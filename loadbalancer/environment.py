import numpy as np
import random
from collections import deque

class ServerInstance:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.current_load = 0
        self.livestream_load = 0
        self.vod_load = 0
        self.response_time = 20  # base response time in ms
    
    def add_load(self, load_amount, is_livestream):
        if self.current_load + load_amount > self.capacity:
            return False
        
        self.current_load += load_amount
        if is_livestream:
            self.livestream_load += load_amount
        else:
            self.vod_load += load_amount
            
        # Recalculate response time based on load
        self.response_time = 20 + (self.current_load / self.capacity) * 180
        return True
    
    def reduce_load(self, amount):
        self.current_load = max(0, self.current_load - amount)
        # Simplistic load reduction that doesn't account for which type
        reduction_ratio = amount / (self.livestream_load + self.vod_load + 0.001)
        self.livestream_load = max(0, self.livestream_load - self.livestream_load * reduction_ratio)
        self.vod_load = max(0, self.vod_load - self.vod_load * reduction_ratio)
        
        # Recalculate response time
        self.response_time = 20 + (self.current_load / self.capacity) * 180

class AWSEnvironment:
    def __init__(self, num_servers=5, max_queue_size=100):
        self.servers = [ServerInstance() for _ in range(num_servers)]
        self.max_servers = 10  # Maximum number of servers we can scale to
        self.livestream_queue = deque(maxlen=max_queue_size)
        self.vod_queue = deque(maxlen=max_queue_size)
        self.dropped_requests = 0
        self.total_requests = 0
        self.livestream_response_times = []
        self.vod_response_times = []
    
    def generate_traffic(self, timestep):
        base_livestream = 5 + 3 * np.sin(timestep / 50)
        base_vod = 8 + 2 * np.sin(timestep / 80 + 1)
        
        # Add randomness
        livestream_requests = max(0, int(base_livestream + np.random.normal(0, 2)))
        vod_requests = max(0, int(base_vod + np.random.normal(0, 3)))
        
        # Add requests to queues
        for _ in range(livestream_requests):
            # (request_size, is_livestream, time_entered)
            self.livestream_queue.append((random.randint(1, 5), True, timestep))
        
        for _ in range(vod_requests):
            self.vod_queue.append((random.randint(1, 3), False, timestep))
        
        self.total_requests += livestream_requests + vod_requests
    
    def get_state(self):
        # Create a state representation with fixed length for the RL agent
        max_servers = self.max_servers  # Maximum possible number of servers
        
        # Initialize fixed-size arrays
        server_loads = [0.0] * max_servers
        server_livestream = [0.0] * max_servers
        server_vod = [0.0] * max_servers
        
        # Fill in values for active servers
        for i, server in enumerate(self.servers):
            if i < max_servers:
                server_loads[i] = server.current_load / server.capacity
                server_livestream[i] = server.livestream_load / server.capacity
                server_vod[i] = server.vod_load / server.capacity
        
        livestream_queue_size = len(self.livestream_queue) 
        vod_queue_size = len(self.vod_queue)
        
        # Combine all state components
        state = server_loads + server_livestream + server_vod + [
            livestream_queue_size / self.livestream_queue.maxlen,
            vod_queue_size / self.vod_queue.maxlen
        ]
        return np.array(state)
    
    def step(self, action_server_idx, action_is_livestream):
        reward = 0
        
        # Add server if needed and if action indicates
        if action_server_idx >= len(self.servers) and len(self.servers) < self.max_servers:
            self.servers.append(ServerInstance())
            reward -= 10  # Cost of adding a server
        
        # Select actual server index (capped at available servers)
        server_idx = min(action_server_idx, len(self.servers) - 1)
        
        # Process from appropriate queue based on action
        queue = self.livestream_queue if action_is_livestream else self.vod_queue
        
        if queue:
            request = queue.popleft()
            request_size, is_livestream, time_entered = request
            
            # Try to add load to selected server
            success = self.servers[server_idx].add_load(request_size, is_livestream)
            
            if success:
                # Calculate waiting time (current time - time entered queue)
                waiting_time = 0  # In a real system, this would be current_time - time_entered
                
                # Calculate response time including server processing
                response_time = waiting_time + self.servers[server_idx].response_time
                
                # Store response times for metrics
                if is_livestream:
                    self.livestream_response_times.append(response_time)
                    # Reduce the difference between livestream and VOD rewards
                    reward += 3 * (300 - min(response_time, 300)) / 300  # Was 5, now 3
                else:
                    self.vod_response_times.append(response_time)
                    # Increase VOD reward
                    reward += 2 * (500 - min(response_time, 500)) / 500  # Keep as 2
            else:
                # Put back in queue or drop if queue full
                if len(queue) < queue.maxlen:
                    queue.appendleft(request)
                    reward -= 1  # Penalty for failing to process
                else:
                    self.dropped_requests += 1
                    # Bigger penalty for dropping livestream
                    reward -= 20 if is_livestream else 10
        
        # Add service level guarantee penalties
        vod_sla_violation = len(self.vod_queue) > self.vod_queue.maxlen * 0.9
        if vod_sla_violation:
            reward -= 5  # Stronger penalty for SLA violations

        # Add a penalty for ignored queues
        if len(self.vod_queue) > self.vod_queue.maxlen * 0.8:
            reward -= 0.5  # Small continuous penalty for letting VOD queue grow

        # Natural reduction in load over time (requests being completed)
        for server in self.servers:
            server.reduce_load(random.uniform(1, 5))
        
        # Return new state, reward, and whether we're in a terminal state (always False for this continuous task)
        return self.get_state(), reward, False, {}
    
    def get_metrics(self):
        # Calculate performance metrics
        avg_livestream_time = np.mean(self.livestream_response_times) if self.livestream_response_times else 0
        avg_vod_time = np.mean(self.vod_response_times) if self.vod_response_times else 0
        drop_rate = self.dropped_requests / max(1, self.total_requests)
        
        # Calculate fairness metric (Jain's fairness index)
        if avg_livestream_time > 0 and avg_vod_time > 0:
            sum_squared = (avg_livestream_time + avg_vod_time)**2
            sum_of_squares = avg_livestream_time**2 + avg_vod_time**2
            fairness = sum_squared / (2 * sum_of_squares) if sum_of_squares > 0 else 0
        else:
            fairness = 0
        
        return {
            "avg_livestream_response": avg_livestream_time,
            "avg_vod_response": avg_vod_time,
            "fairness": fairness,
            "drop_rate": drop_rate,
            "num_servers": len(self.servers),
            "livestream_queue": len(self.livestream_queue),
            "vod_queue": len(self.vod_queue)
        }