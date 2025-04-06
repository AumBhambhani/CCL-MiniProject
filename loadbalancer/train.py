from multi_app_balancer import MultiAppLoadBalancer
from multi_app_environment import MultiAppEnvironment
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_multi_app_metrics(rewards, metrics_history=None, filename="multi_app_plot.png"):
    if not metrics_history:
        plt.figure(figsize=(8, 6))
        plt.plot(rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig(filename)
        plt.close()
        return
        
    app_names = [k.split('_')[0] for k in metrics_history[0].keys() if k.endswith('_response') and 'user_profiles' not in k]
    episodes = range(len(metrics_history))
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(2, 2, 2)
    for app in app_names:
        response_times = [m[f"{app}_response"] for m in metrics_history]
        plt.plot(episodes, response_times, label=app.capitalize())
    plt.title('Response Times by Application')
    plt.xlabel('Episode')
    plt.ylabel('Response Time (ms)')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    for app in app_names:
        drop_rates = [m[f"{app}_drop_rate"] for m in metrics_history]
        plt.plot(episodes, drop_rates, label=app.capitalize())
    plt.title('Drop Rates by Application')
    plt.xlabel('Episode')
    plt.ylabel('Drop Rate')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    for app in app_names:
        queue_sizes = [m[f"{app}_queue"] for m in metrics_history]
        plt.plot(episodes, queue_sizes, label=app.capitalize())
    plt.title('Queue Sizes by Application')
    plt.xlabel('Episode')
    plt.ylabel('Queue Size')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    os.makedirs("results", exist_ok=True)
    
    load_balancer = MultiAppLoadBalancer(initial_servers=8)
    load_balancer.env = MultiAppEnvironment(num_servers=8, seed=42)
    
    print("Starting training...")
    rewards = load_balancer.train(episodes=10, timesteps_per_episode=200)
    
    load_balancer.save("results/multi_app_model")
    
    plot_multi_app_metrics(rewards, filename="results/multi_app_training.png")
    
    print("\nStarting evaluation...")
    eval_balancer = MultiAppLoadBalancer(initial_servers=8)
    eval_balancer.env = MultiAppEnvironment(num_servers=8, seed=42)
    eval_balancer.load("results/multi_app_model")
    metrics_history = eval_balancer.evaluate(timesteps=1000)
    
    plot_multi_app_metrics([], metrics_history, filename="results/multi_app_evaluation.png")
    
    print("Training and evaluation complete!")

if __name__ == "__main__":
    main()