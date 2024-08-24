import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

import numpy as np

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer with 192 neurons
        return x
    
def to_tensor(data, dtype):
    return torch.tensor(np.array(data), dtype=dtype)
    
class QTrainer:
    def __init__(self, model: QNetwork, lr: float, gamma: float) -> None:
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss(reduction='mean')

    def step(self, state, action, reward, next_state, done):
        state = to_tensor(state, dtype=torch.float)
        next_state = to_tensor(next_state, dtype=torch.float)
        action = to_tensor(action, dtype=torch.long)
        reward = to_tensor(reward, dtype=torch.float)
        done = to_tensor(done, dtype=torch.bool)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        #print(action)

        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            #i = torch.argmax(action).item()
            i = action[idx].item()

            if i > len(target[idx]):
                continue

            target[idx][i] = Q_new

        # q_values = self.model(state)
        # next_q_values = self.model(next_state)

        # q_value = q_values.gather(1, action)
        # next_q_value = next_q_values.max(1)[0]
        # excepted_q_value = reward + self.gamma * next_q_value * (1 - done)

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()