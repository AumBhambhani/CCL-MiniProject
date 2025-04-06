from environment import AWSEnvironment
from dqn_agent import DQNAgent
import numpy as np
import time

class RLLoadBalancer:
    def __init__(self, initial_servers=5, max_action_servers=10):
        self.env = AWSEnvironment(num_servers=initial_servers)
        
        # State size: server loads + livestream loads + vod loads + queue sizes
        state_size = 3 * max_action_servers + 2  # 3 metrics per server + 2 queue metrics
        
        # Action space: (server_idx, is_livestream)
        # server_idx can be 0 to max_action_servers-1
        # is_livestream can be True or False
        action_size = max_action_servers * 2  # For each server, we can choose livestream or VOD
        
        self.agent = DQNAgent(state_size, action_size)
        self.max_action_servers = max_action_servers
        self.training_mode = True
    
    def set_training(self, is_training):
        self.training_mode = is_training
    
    def process_timestep(self, timestep):
        # Generate new traffic
        self.env.generate_traffic(timestep)
        
        # Get current state
        state = self.env.get_state()
        
        # Process requests from both queues
        num_requests = len(self.env.livestream_queue) + len(self.env.vod_queue)
        
        total_reward = 0
        
        # Add VOD starvation prevention
        vod_queue_size = len(self.env.vod_queue)
        livestream_queue_size = len(self.env.livestream_queue)
        
        # Force processing some VOD if queue is getting too full
        force_vod = vod_queue_size > 90 and timestep % 10 == 0
        
        # Process multiple requests in each timestep
        for _ in range(min(num_requests, 10)):
            state = self.env.get_state()
            
            # Get action from agent
            action = self.agent.act(state, training=self.training_mode)
            server_idx, is_livestream = action
            
            # Override action if needed to prevent VOD starvation
            if force_vod and vod_queue_size > 0:
                is_livestream = False
                force_vod = False  # Only force once per timestep
            
            # Take action in environment
            next_state, reward, done, _ = self.env.step(server_idx, is_livestream)
            total_reward += reward
            
            # Store experience in replay memory
            if self.training_mode:
                self.agent.remember(state, action, reward, next_state, done)
        
        # Train agent on past experiences
        if self.training_mode:
            self.agent.replay()
        
        return total_reward, self.env.get_metrics()
    
    def train(self, episodes=1000, timesteps_per_episode=100, update_freq=10, save_freq=100):
        rewards_history = []
        
        for episode in range(episodes):
            # Reset environment
            self.env = AWSEnvironment(num_servers=5)
            episode_reward = 0
            
            for t in range(timesteps_per_episode):
                reward, metrics = self.process_timestep(t)
                episode_reward += reward
                
                # Periodically update target network
                if t % update_freq == 0:
                    self.agent.update_target_model()
            
            rewards_history.append(episode_reward)
            
            # Print episode summary
            print(f"Episode: {episode+1}/{episodes}, Reward: {episode_reward:.2f}, Epsilon: {self.agent.epsilon:.4f}")
            print(f"Metrics: {metrics}")
            
            # Periodically save model
            if (episode + 1) % save_freq == 0:
                self.agent.save(f"dqn_loadbalancer_ep{episode+1}.h5")
                
        return rewards_history
    
    def evaluate(self, timesteps=1000):
        # Set to evaluation mode
        self.set_training(False)
        
        # Reset environment
        self.env = AWSEnvironment(num_servers=5)
        
        metrics_history = []
        reward_history = []
        
        for t in range(timesteps):
            reward, metrics = self.process_timestep(t)
            reward_history.append(reward)
            metrics_history.append(metrics)
            
            if t % 100 == 0:
                print(f"Timestep {t}/{timesteps}, Reward: {reward:.2f}")
                print(f"Metrics: {metrics}")
        
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