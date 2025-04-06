import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQNNetwork(state_size, action_size).to(self.device)
        self.target_model = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_model()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
    
    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            if hasattr(self, 'action_style') and self.action_style == 'flat':
                return random.randint(0, self.action_size - 1)
            else:
                server_idx = random.randint(0, (self.action_size // 2) - 1)
                is_livestream = random.choice([True, False])
                return (server_idx, is_livestream)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_values = self.model(state_tensor)
            
            if hasattr(self, 'action_style') and self.action_style == 'flat':
                return torch.argmax(act_values).item()
            else:
                action_idx = torch.argmax(act_values).item()
                server_idx = action_idx // 2
                is_livestream = (action_idx % 2) == 0
                return (server_idx, is_livestream)
    
    def replay(self):
        if self.memory.size() < self.batch_size:
            return
        
        minibatch = self.memory.sample(self.batch_size)
        batch_loss = 0
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            reward_tensor = torch.FloatTensor([reward]).to(self.device)
            done_tensor = torch.FloatTensor([float(done)]).to(self.device)
            current_q = self.model(state_tensor)
            
            with torch.no_grad():
                next_q = self.target_model(next_state_tensor)
                max_next_q = torch.max(next_q, dim=1)[0]
                target_q = reward_tensor + (1 - done_tensor) * self.gamma * max_next_q
            
            if isinstance(action, tuple):
                server_idx, is_livestream = action
                action_idx = server_idx * 2 + (0 if is_livestream else 1)
            else:
                action_idx = action
            
            action_tensor = torch.LongTensor([[action_idx]]).to(self.device)
            q_value = current_q.gather(1, action_tensor)
            loss = self.criterion(q_value, target_q.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_loss += loss.item()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return batch_loss / self.batch_size
    
    def load(self, name):
        self.model.load_state_dict(torch.load(name + "_model.pth"))
        self.target_model.load_state_dict(torch.load(name + "_target.pth"))
        self.epsilon = self.epsilon_min
        self.model.eval()
        self.target_model.eval()
    
    def save(self, name):
        torch.save(self.model.state_dict(), name + "_model.pth")
        torch.save(self.target_model.state_dict(), name + "_target.pth")