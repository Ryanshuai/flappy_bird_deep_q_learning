import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import cv2
import pickle
import os
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralNet(nn.Module):
    def __init__(self, load_path=None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()

        self.fc1 = nn.Linear(2048, 256)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

        self.load(load_path)

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

    def load(self, load_path=None):
        try:
            with open(load_path, "rb") as f:
                state_dict = torch.load(f, map_location=DEVICE)
            self.load_state_dict(state_dict)
            print("Model loaded from {}".format(load_path))
        except FileNotFoundError:
            print("No model parameter found. Starting from scratch.")


class Memory:
    def __init__(self, action_size, load_folder=None):
        self.action_size = action_size
        self.mem = deque(maxlen=10_000)
        if load_folder:
            self.from_disk(load_folder)

    def save(self, state, action, reward, next_state, done):
        state = state.transpose((2, 0, 1))
        next_state = next_state.transpose((2, 0, 1))
        action = np.eye(self.action_size)[action]
        self.mem.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        mem_sample = random.sample(self.mem, batch_size)
        state, action, reward, next_state, done = zip(*mem_sample)
        state = torch.from_numpy(np.array(state)).float().cuda()
        action = torch.from_numpy(np.array(action)).bool().cuda()
        reward = torch.from_numpy(np.array(reward)).float().cuda()
        next_state = torch.from_numpy(np.array(next_state)).float().cuda()
        done = torch.from_numpy(np.array(done)).int().cuda()
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.mem)

    def to_disk(self, save_name):
        with open(save_name, "wb") as f:
            pickle.dump(self.mem, f)

    def from_disk(self, load_name):
        try:
            with open(load_name, "rb") as f:
                self.mem = pickle.load(f)
        except FileNotFoundError:
            print("No memory found. Starting from scratch.")


class DQN_Agent:
    def __init__(self, action_size, train=True):
        self.action_size = action_size
        self.memory = Memory(action_size, load_folder="memory")

        self.batch_size = 32
        self.gamma = 0.99

        load_path = "pths/3_4.pth"
        self.model = NeuralNet(load_path).to(DEVICE)
        self.model_fixed = NeuralNet(load_path).to(DEVICE)
        self.writer = SummaryWriter("runs/exp1")

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        if train:
            self.random_action_probability = 0.3
            self.min_random_action_probability = 0.01
            self.random_action_probability_decay = 0.99_999
        else:
            self.random_action_probability = 0.0
            self.min_random_action_probability = 0.0
            self.random_action_probability_decay = 0.0

        self.observe_counter = 0
        self.learn_counter = 0

    def observe(self, state, action, reward, next_state, done):
        state = cv2.resize(state, (128, 64))
        next_state = cv2.resize(next_state, (128, 64))
        self.memory.save(state, action, reward, next_state, done)
        self.writer.add_scalar("Reward/reward", reward, self.learn_counter)

    def act(self, state):
        self.random_action_probability = max(self.random_action_probability * self.random_action_probability_decay,
                                             self.min_random_action_probability)

        if np.random.uniform(0, 1) < self.random_action_probability:
            p = np.array([0.9, 0.1])
            index = np.random.choice([0, 1], p=p.ravel())
            return index
        else:
            state = cv2.resize(state, (128, 64))
            state = state.transpose((2, 0, 1))
            state = torch.from_numpy(np.array(state)).float().cuda().unsqueeze(0)
            max_idx = torch.argmax(self.model_fixed(state)).item()
            return max_idx

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        q_fixed_next_state = self.model_fixed(next_state).detach()
        q_next_state = self.model(next_state).detach()
        next_q_value = q_fixed_next_state.gather(1, torch.max(q_next_state, 1)[1].unsqueeze(1)).squeeze(1)

        estimate_reward = reward + self.gamma * next_q_value * (1 - done)
        loss = self.loss_function(estimate_reward, self.model(state).masked_select(action))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_counter += 1

        if self.learn_counter % 10 == 0:
            self.model_fixed.load_state_dict(self.model.state_dict())

        if self.learn_counter % 10_000 == 0:
            self.memory.to_disk("memory")

        if self.learn_counter % 10_000 == 0:
            torch.save(self.model.state_dict(), "pths" + os.sep + f"1_{self.learn_counter // 10_000}" + ".pth")

        self.writer.add_scalar('Loss', loss.item(), self.learn_counter)
        self.writer.add_scalar('Random Action Probability', self.random_action_probability, self.learn_counter)
        self.writer.add_scalar('Reward/q_value', torch.max(self.model(state)).item(), self.learn_counter)
