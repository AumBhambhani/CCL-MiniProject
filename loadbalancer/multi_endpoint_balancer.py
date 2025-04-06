import numpy as np
from environment import AWSEnvironment
from dqn_agent import DQNAgent

class MultiEndpointLoadBalancer:
    def __init__(self, initial_servers=8):
        # Create specialized server pools
        self.livestream_servers = initial_servers // 2  # Dedicated livestream servers
        self.vod_servers = initial_servers // 4        # Dedicated VOD servers
        self.flex_servers = initial_servers - self.livestream_servers - self.vod_servers  # Flex capacity
        
        self.env = AWSEnvironment(num_servers=initial_servers)
        
        # Create separate agents for different decisions
        # 1. Resource allocation agent (how many servers in each pool)
        alloc_state_size = 5  # queue sizes + current allocation
        alloc_action_size = 10  # Different allocation combinations
        self.allocation_agent = DQNAgent(alloc_state_size, alloc_action_size)
        
        # 2. Request routing agent (which specific server for each request)
        route_state_size = 3 * initial_servers + 2
        route_action_size = initial_servers * 2
        self.routing_agent = DQNAgent(route_state_size, route_action_size)
    
    def process_timestep(self, timestep):
        # Generate new traffic
        self.env.generate_traffic(timestep)
        
        # 1. First, make resource allocation decision (every 10 timesteps)
        if timestep % 10 == 0:
            self._reallocate_resources()
            
        # 2. Then route individual requests
        total_reward = self._route_requests()
        
        return total_reward, self.env.get_metrics()
    
    def _reallocate_resources(self):
        """Decide how to allocate servers between different traffic types"""
        # Simplified state for allocation decisions
        ls_queue = len(self.env.livestream_queue) / self.env.livestream_queue.maxlen
        vod_queue = len(self.env.vod_queue) / self.env.vod_queue.maxlen
        
        state = np.array([
            ls_queue, 
            vod_queue,
            self.livestream_servers / len(self.env.servers),
            self.vod_servers / len(self.env.servers),
            self.flex_servers / len(self.env.servers)
        ])
        
        # Get allocation action (0-9 representing different allocation patterns)
        action = self.allocation_agent.act(state)
        
        # Map action to concrete allocation
        # Example mapping (can be adjusted):
        allocations = [
            (0.6, 0.3, 0.1),  # 60% livestream, 30% VOD, 10% flex
            (0.5, 0.3, 0.2),
            (0.5, 0.4, 0.1),
            (0.4, 0.4, 0.2),
            (0.7, 0.2, 0.1),
            (0.3, 0.6, 0.1),
            (0.4, 0.5, 0.1),
            (0.6, 0.2, 0.2),
            (0.3, 0.3, 0.4),
            (0.4, 0.3, 0.3)
        ]
        
        ls_ratio, vod_ratio, flex_ratio = allocations[action]
        total_servers = len(self.env.servers)
        
        # Update allocations (ensure at least 1 server per type)
        self.livestream_servers = max(1, int(ls_ratio * total_servers))
        self.vod_servers = max(1, int(vod_ratio * total_servers))
        self.flex_servers = total_servers - self.livestream_servers - self.vod_servers
    
    def _route_requests(self):
        """Route individual requests to specific servers"""
        num_requests = len(self.env.livestream_queue) + len(self.env.vod_queue)
        total_reward = 0
        
        # Process a batch of requests
        for _ in range(min(num_requests, 15)):
            state = self.env.get_state()
            
            # Check queue status
            ls_queue_size = len(self.env.livestream_queue)
            vod_queue_size = len(self.env.vod_queue)
            
            # Pick which type to process based on both priority and starvation prevention
            process_livestream = True
            
            # Logic to prevent starvation
            if ls_queue_size == 0:
                process_livestream = False
            elif vod_queue_size > 0:
                # Probabilistic decision based on queue sizes and priorities
                ls_priority = 2.0  # Livestream priority multiplier
                total = (ls_queue_size * ls_priority) + vod_queue_size
                ls_prob = (ls_queue_size * ls_priority) / total
                process_livestream = np.random.random() < ls_prob
            
            # Get server assignment from routing agent
            server_idx, _ = self.routing_agent.act(state)
            
            # Determine appropriate server pool based on request type
            if process_livestream:
                # Use livestream or flex servers
                if server_idx < self.livestream_servers:
                    pass  # Already in correct range
                else:
                    # Use flex servers for overflow
                    server_idx = self.livestream_servers + self.vod_servers + (server_idx % self.flex_servers)
            else:
                # Use VOD servers or flex servers
                server_idx = self.livestream_servers + (server_idx % (self.vod_servers + self.flex_servers))
                if server_idx >= self.livestream_servers + self.vod_servers:
                    # Adjust index to use flex servers
                    server_idx = self.livestream_servers + self.vod_servers + (server_idx % self.flex_servers)
            
            # Execute the step
            next_state, reward, done, _ = self.env.step(server_idx, process_livestream)
            total_reward += reward
            
            # Remember experience
            self.routing_agent.remember(state, (server_idx, process_livestream), reward, next_state, done)
        
        # Train agents
        self.routing_agent.replay()
        
        return total_reward

    def set_training(self, is_training):
        """Set whether agents are in training mode"""
        self.training_mode = is_training
    
    def train(self, episodes=1000, timesteps_per_episode=100, update_freq=10, save_freq=100):
        """Train both agents together"""
        rewards_history = []
        
        for episode in range(episodes):
            # Reset environment
            self.env = AWSEnvironment(num_servers=8)  # Initial servers for multi-endpoint
            self.livestream_servers = 4  # Reset server allocation
            self.vod_servers = 2
            self.flex_servers = 2
            episode_reward = 0
            
            for t in range(timesteps_per_episode):
                reward, metrics = self.process_timestep(t)
                episode_reward += reward
                
                # Periodically update target networks
                if t % update_freq == 0:
                    self.allocation_agent.update_target_model()
                    self.routing_agent.update_target_model()
            
            rewards_history.append(episode_reward)
            
            # Print episode summary
            print(f"Episode: {episode+1}/{episodes}, Reward: {episode_reward:.2f}")
            print(f"Metrics: {metrics}")
            print(f"Server allocation: LS={self.livestream_servers}, VOD={self.vod_servers}, Flex={self.flex_servers}")
            
            # Periodically save models
            if (episode + 1) % save_freq == 0:
                self.save(f"multi_lb_ep{episode+1}")
                
        return rewards_history

    def evaluate(self, timesteps=1000):
        """Evaluate the trained agents"""
        # Set to evaluation mode
        self.set_training(False)
        
        # Reset environment
        self.env = AWSEnvironment(num_servers=8)
        self.livestream_servers = 4
        self.vod_servers = 2
        self.flex_servers = 2
        
        metrics_history = []
        reward_history = []
        
        for t in range(timesteps):
            reward, metrics = self.process_timestep(t)
            reward_history.append(reward)
            metrics_history.append(metrics)
            
            if t % 100 == 0:
                print(f"Timestep {t}/{timesteps}, Reward: {reward:.2f}")
                print(f"Metrics: {metrics}")
                print(f"Server allocation: LS={self.livestream_servers}, VOD={self.vod_servers}, Flex={self.flex_servers}")
        
        # Print summary
        avg_reward = np.mean(reward_history)
        avg_livestream_resp = np.mean([m["avg_livestream_response"] for m in metrics_history])
        avg_vod_resp = np.mean([m["avg_vod_response"] for m in metrics_history])
        avg_drop_rate = np.mean([m["drop_rate"] for m in metrics_history])
        
        print("\nEvaluation Summary:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Livestream Response Time: {avg_livestream_resp:.2f}ms")
        print(f"Average VOD Response Time: {avg_vod_resp:.2f}ms")
        print(f"Average Drop Rate: {avg_drop_rate:.4f}")
        
        return metrics_history

    def save(self, name):
        """Save both agent models"""
        self.allocation_agent.save(name + "_allocation")
        self.routing_agent.save(name + "_routing")

    def load(self, name):
        """Load both agent models"""
        self.allocation_agent.load(name + "_allocation")
        self.routing_agent.load(name + "_routing")