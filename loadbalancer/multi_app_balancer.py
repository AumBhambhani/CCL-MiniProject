import numpy as np
import torch
from multi_app_environment import MultiAppEnvironment
from dqn_agent import DQNAgent
import random

class MultiAppLoadBalancer:
    def __init__(self, initial_servers=8):
        self.env = MultiAppEnvironment(num_servers=initial_servers)
        self.app_names = list(self.env.applications.keys())
        self.training_mode = True
        num_servers = self.env.max_servers
        num_tracked_apps = 3
        num_apps = len(self.app_names)
        state_size = num_servers + (num_tracked_apps * num_servers) + num_apps
        action_size = num_servers * num_apps
        self.agent = DQNAgent(state_size, action_size)
        self.agent.action_style = 'flat'

    def set_training(self, is_training):
        self.training_mode = is_training
        
    def process_timestep(self, timestep):
        self.env.generate_traffic(timestep)
        queue_sizes = {app: len(self.env.queues[app]) for app in self.app_names}
        if timestep % 100 == 0 and not self.training_mode:
            print(f"Queue sizes: {queue_sizes}")
        user_queue = len(self.env.queues.get('user_profiles', []))
        user_processed = 0
        total_requests = sum(queue_sizes.values())
        total_reward = 0
        for _ in range(min(total_requests, 20)):
            state = self.env.get_state()
            action_tuple = self.agent.act(state, training=self.training_mode)
            if isinstance(action_tuple, tuple):
                server_idx, is_livestream = action_tuple
                if is_livestream and "livestream" in self.app_names:
                    app_name = "livestream"
                elif not is_livestream and "vod" in self.app_names:
                    app_name = "vod"
                else:
                    app_name = self.app_names[0]
            else:
                server_idx = action_tuple // len(self.app_names)
                app_idx = action_tuple % len(self.app_names)
                app_name = self.app_names[app_idx]
            if len(self.env.queues[app_name]) == 0:
                non_empty_apps = [app for app in self.app_names if len(self.env.queues[app]) > 0]
                if non_empty_apps:
                    app_name = max(non_empty_apps, key=lambda a: self.env.applications[a].priority)
            next_state, reward, done, _ = self.env.step(server_idx, app_name)
            total_reward += reward
            if self.training_mode:
                app_idx = self.app_names.index(app_name)
                flat_action = server_idx * len(self.app_names) + app_idx
                self.agent.remember(state, flat_action, reward, next_state, done)
            if app_name == 'user_profiles':
                user_processed += 1
        if self.training_mode:
            self.agent.replay()
        if not self.training_mode and timestep % 100 == 0:
            print(f"Debug - User profiles: Queue={user_queue}, Processed={user_processed}")
        return total_reward, self.env.get_metrics()
    
    def train(self, episodes=50, timesteps_per_episode=200, update_freq=10):
        rewards_history = []
        for episode in range(episodes):
            self.env = MultiAppEnvironment(num_servers=8)
            episode_reward = 0
            for t in range(timesteps_per_episode):
                reward, metrics = self.process_timestep(t)
                episode_reward += reward
                if t % update_freq == 0:
                    self.agent.update_target_model()
            rewards_history.append(episode_reward)
            print(f"Episode: {episode+1}/{episodes}, Reward: {episode_reward:.2f}")
            print(f"Metrics: {', '.join([f'{k}: {v:.2f}' for k, v in metrics.items() if '_response' in k])}")
        return rewards_history
    
    def evaluate(self, timesteps=1000):
        self.set_training(False)
        self.env = MultiAppEnvironment(num_servers=8)
        metrics_history = []
        reward_history = []
        for t in range(timesteps):
            reward, metrics = self.process_timestep(t)
            reward_history.append(reward)
            metrics_history.append(metrics)
            if t % 100 == 0:
                print(f"Timestep {t}/{timesteps}, Reward: {reward:.2f}")
                print(f"Response times: {', '.join([f'{k}: {v:.2f}' for k, v in metrics.items() if '_response' in k])}")
        avg_reward = np.mean(reward_history)
        print("\nEvaluation Summary:")
        print(f"Average Reward: {avg_reward:.2f}")
        for app_name in self.app_names:
            avg_resp = np.mean([m[f"{app_name}_response"] for m in metrics_history])
            avg_drop = np.mean([m[f"{app_name}_drop_rate"] for m in metrics_history])
            print(f"{app_name.capitalize()} - Avg Response: {avg_resp:.2f}ms, Drop Rate: {avg_drop:.4f}")
        return metrics_history
    
    def save(self, name):
        self.agent.save(name)

    def load(self, name):
        self.agent.load(name)
        self.set_training(False)

    def warm_up_replay(self, timesteps=1000):
        print("Warming up replay buffer with random experiences...")
        warm_env = MultiAppEnvironment(num_servers=8, seed=42)
        for t in range(timesteps):
            warm_env.generate_traffic(t)
            state = warm_env.get_state()
            for _ in range(10):
                server_idx = random.randint(0, warm_env.max_servers-1)
                app_name = random.choice(self.app_names)
                next_state, reward, done, _ = warm_env.step(server_idx, app_name)
                app_idx = self.app_names.index(app_name)
                flat_action = server_idx * len(self.app_names) + app_idx
                self.agent.remember(state, flat_action, reward, next_state, done)
                state = next_state
        print(f"Replay buffer warmed up with {self.agent.memory.size()} experiences")