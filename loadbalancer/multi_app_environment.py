import numpy as np
import random
from collections import deque
import torch

class ApplicationType:
    """Defines a specific application with its characteristics"""
    def __init__(self, name, priority, latency_requirement, 
                 resource_intensity, traffic_pattern="steady"):
        self.name = name
        self.priority = priority
        self.latency_requirement = latency_requirement
        self.resource_intensity = resource_intensity
        self.traffic_pattern = traffic_pattern
        
        self.response_times = []
        self.dropped_requests = 0
        self.processed_requests = 0

class MultiAppEnvironment:
    def __init__(self, num_servers=8, max_queue_size=100, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        self.servers = [ServerInstance() for _ in range(num_servers)]
        self.max_servers = 20
        self.timestep = 0
        self.applications = {
            "livestream": ApplicationType(
                name="livestream", 
                priority=10, 
                latency_requirement=100, 
                resource_intensity=8, 
                traffic_pattern="periodic"
            ),
            "vod": ApplicationType(
                name="vod", 
                priority=6, 
                latency_requirement=500, 
                resource_intensity=7, 
                traffic_pattern="steady"
            ),
            "chat": ApplicationType(
                name="chat", 
                priority=9, 
                latency_requirement=150, 
                resource_intensity=3, 
                traffic_pattern="bursty"
            ),
            "user_profiles": ApplicationType(
                name="user_profiles", 
                priority=5, 
                latency_requirement=300, 
                resource_intensity=2, 
                traffic_pattern="steady"
            ),
            "analytics": ApplicationType(
                name="analytics", 
                priority=2, 
                latency_requirement=5000, 
                resource_intensity=10, 
                traffic_pattern="periodic"
            )
        }
        self.queues = {app_name: deque(maxlen=max_queue_size) 
                      for app_name in self.applications.keys()}
        
    def generate_traffic(self, timestep):
        self.timestep = timestep
        for app_name, app in self.applications.items():
            base_traffic = self._get_base_traffic(app)
            traffic_amount = max(0, int(base_traffic + np.random.normal(0, base_traffic/4)))
            for _ in range(traffic_amount):
                request_size = random.randint(1, app.resource_intensity)
                self.queues[app_name].append((request_size, app_name, timestep))
                
    def _get_base_traffic(self, app):
        if app.traffic_pattern == "steady":
            return 5 + app.priority/3
        elif app.traffic_pattern == "periodic":
            if app.name == "livestream":
                day_cycle = np.sin(self.timestep / 288 * 2 * np.pi)
                week_cycle = np.sin(self.timestep / 2016 * 2 * np.pi)
                return 8 + 6 * max(0, day_cycle) + 4 * max(0, week_cycle)
            else:
                return 5 + 5 * np.sin(self.timestep / 100 + hash(app.name) % 10)
        elif app.traffic_pattern == "bursty":
            if random.random() < 0.05:
                return 20 + app.priority
            else:
                return 3 + app.priority/5
        return 5
    
    def step(self, server_idx, app_name):
        if app_name not in self.queues or not self.queues[app_name]:
            return self.get_state(), 0, False, {}
        app = self.applications[app_name]
        request = self.queues[app_name].popleft()
        request_size, _, time_entered = request
        if server_idx >= len(self.servers):
            if len(self.servers) < self.max_servers:
                self.servers.append(ServerInstance())
                server_idx = len(self.servers) - 1
                reward = -15
            else:
                server_idx = len(self.servers) - 1
                reward = -5
        success = self.servers[server_idx].add_load(request_size, app_name)
        if success:
            waiting_time = self.timestep - time_entered
            response_time = waiting_time + self.servers[server_idx].response_time
            app.response_times.append(response_time)
            app.processed_requests += 1
            latency_satisfaction = max(0, 1 - (response_time / app.latency_requirement))
            reward = app.priority * latency_satisfaction
        else:
            if len(self.queues[app_name]) < self.queues[app_name].maxlen:
                self.queues[app_name].appendleft(request)
                reward = -1
            else:
                app.dropped_requests += 1
                reward = -2 * app.priority
        for server in self.servers:
            server.reduce_load(random.uniform(1, 5))
        return self.get_state(), reward, False, {}
    
    def get_state(self):
        server_loads = [s.current_load / s.capacity for s in self.servers]
        server_loads += [0] * (self.max_servers - len(server_loads))
        app_names = ["livestream", "vod", "chat"]
        app_loads = []
        for app_name in app_names:
            loads = [s.app_loads.get(app_name, 0) / s.capacity for s in self.servers]
            loads += [0] * (self.max_servers - len(loads))
            app_loads.extend(loads)
        queue_states = [len(self.queues[app_name]) / self.queues[app_name].maxlen 
                        for app_name in self.applications.keys()]
        state = server_loads + app_loads + queue_states
        return np.array(state)
    
    def get_metrics(self):
        metrics = {}
        for app_name, app in self.applications.items():
            avg_response = np.mean(app.response_times[-100:]) if app.response_times else 0
            drop_rate = app.dropped_requests / max(1, app.dropped_requests + app.processed_requests)
            metrics[f"{app_name}_response"] = avg_response
            metrics[f"{app_name}_drop_rate"] = drop_rate
            metrics[f"{app_name}_queue"] = len(self.queues[app_name])
        metrics["num_servers"] = len(self.servers)
        response_times = [metrics[f"{app_name}_response"] for app_name in self.applications.keys()]
        if all(rt > 0 for rt in response_times):
            n = len(response_times)
            sum_squared = sum(response_times)**2
            sum_of_squares = sum(rt**2 for rt in response_times)
            fairness = sum_squared / (n * sum_of_squares) if sum_of_squares > 0 else 0
            metrics["fairness"] = fairness
        else:
            metrics["fairness"] = 0
        return metrics

class ServerInstance:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.current_load = 0
        self.app_loads = {}
        self.response_time = 20
    
    def add_load(self, load_amount, app_name):
        if self.current_load + load_amount > self.capacity:
            return False
        self.current_load += load_amount
        self.app_loads[app_name] = self.app_loads.get(app_name, 0) + load_amount
        self.response_time = 20 + (self.current_load / self.capacity) * 180
        return True
    
    def reduce_load(self, amount):
        if self.current_load <= 0:
            return
        reduction_ratio = min(1.0, amount / self.current_load) if self.current_load > 0 else 0
        self.current_load = max(0, self.current_load - amount)
        for app in self.app_loads:
            self.app_loads[app] = max(0, self.app_loads[app] - self.app_loads[app] * reduction_ratio)
        self.response_time = 20 + (self.current_load / self.capacity) * 180