ENV_CONFIG = {
    "initial_servers": 5,
    "max_servers": 10,
    "max_queue_size": 100,
    "livestream_priority_factor": 2.5,
}

AGENT_CONFIG = {
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01,
    "batch_size": 64,
    "update_frequency": 10,
}

TRAINING_CONFIG = {
    "episodes": 100,
    "timesteps_per_episode": 200,
    "save_frequency": 20,
}

AWS_CONFIG = {
    "region": "us-east-1",
    "instance_types": {
        "small": "t3.small",
        "medium": "t3.medium",
        "large": "t3.large"
    },
    "scaling_cooldown": 300,
}