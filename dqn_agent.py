import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import os
import random
import numpy as np
from collections import deque

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        self.fc1 = nn.Linear(73728, 256)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))

        x = x.view(x.size(0), -1)
        x = self.relu6(self.fc1(x))
        x = self.fc2(x)

        return x


class Memory:
    def __init__(self, action_size):
        self.action_size = action_size
        self.mem = deque(maxlen=50_000)

    def save(self, state, action, reward, next_state, done):
        state = state.transpose((2, 0, 1))
        next_state = next_state.transpose((2, 0, 1))
        action = np.eye(self.action_size)[action]
        # action = np.eye(self.action_size)[actiosn]
        self.mem.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.mem, batch_size))
        state = torch.from_numpy(np.array(state)).float().cuda()
        action = torch.from_numpy(np.array(action)).bool().cuda()
        reward = torch.from_numpy(np.array(reward)).float().cuda()
        next_state = torch.from_numpy(np.array(next_state)).float().cuda()
        done = torch.from_numpy(np.array(done)).int().cuda()
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.mem)


class DQN_Agent:
    def __init__(self, action_size, train=True):
        self.action_size = action_size

        self.batch_size = 8
        self.gamma = 0.99

        self.memory = Memory(action_size)
        self.model = NeuralNet().to(DEVICE)

        load_pickle = "1_393.pth"
        if os.path.exists("pths/" + load_pickle):
            with open("pths/" + load_pickle, "rb") as f:
                state_dict = torch.load(f, map_location=DEVICE)
            self.model.load_state_dict(state_dict)

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

        if train:
            self.random_action_probability = 0.2
            self.min_random_action_probability = 0.01
            self.random_action_probability_decay = 0.99999
        else:
            self.random_action_probability = 0.0
            self.min_random_action_probability = 0.0
            self.random_action_probability_decay = 0.0

        self.learn_counter = 0
        self.writer = SummaryWriter("runs/exp2")

    def observe(self, state, action, reward, next_state, done):
        self.memory.save(state, action, reward, next_state, done)
        self.writer.add_scalar("reward", reward, self.learn_counter)

    def act(self, state):
        self.random_action_probability = max(self.random_action_probability * self.random_action_probability_decay,
                                             self.min_random_action_probability)

        if np.random.uniform(0, 1) < self.random_action_probability:
            p = np.array([0.9, 0.1])
            index = np.random.choice([0, 1], p=p.ravel())
            return index
        else:
            state = state.transpose((2, 0, 1))
            state = torch.from_numpy(np.array(state)).float().cuda().unsqueeze(0)
            max_idx = torch.argmax(self.model(state)).item()
            self.writer.add_scalar("p_value", self.model(state)[0][max_idx], self.learn_counter)
            return max_idx

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        q_estimate_next_state = self.model(next_state)
        max_q_estimate_next_state = torch.max(q_estimate_next_state, dim=1)[0]
        estimate_reward = reward + self.gamma * max_q_estimate_next_state * (1 - done)
        loss = self.loss_function(estimate_reward, self.model(state).masked_select(action))
        loss.backward()
        # print(loss.item())
        self.optimizer.step()

        self.learn_counter += 1
        if self.learn_counter % 1000 == 0:
            torch.save(self.model.state_dict(), "pths" + os.sep + f"2_{self.learn_counter // 1000}" + ".pth")

        self.writer.add_scalar('Loss', loss.item(), self.learn_counter)
        self.writer.add_scalar('Random Action Probability', self.random_action_probability, self.learn_counter)
