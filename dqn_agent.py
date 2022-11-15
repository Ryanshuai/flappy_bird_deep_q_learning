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
        if load_path and os.path.exists(load_path):
            with open(load_path, "rb") as f:
                state_dict = torch.load(f, map_location=DEVICE)
            self.load_state_dict(state_dict)


class Memory:
    def __init__(self, action_size, load_folder=None):
        self.action_size = action_size

        if load_folder:
            self.from_disk(load_folder)
        else:
            self.mem3 = deque(maxlen=24_000)
            self.mem10 = deque(maxlen=56_000)
            self.mem20 = deque(maxlen=80_000)
            self.mem30 = deque(maxlen=80_000)
            self.mem40 = deque(maxlen=80_000)
            self.mem50 = deque(maxlen=80_000)

        self.temp_mem = []

    def temp_save(self, state, action, reward, next_state, done):
        state = state.transpose((2, 0, 1))
        next_state = next_state.transpose((2, 0, 1))
        action = np.eye(self.action_size)[action]
        self.temp_mem.append((state, action, reward, next_state, done))

    def determine_save(self, sum_reward):
        if sum_reward < 3:
            self.mem3.extend(self.temp_mem)
        elif sum_reward < 10:
            self.mem10.extend(self.temp_mem)
        elif sum_reward < 20:
            self.mem20.extend(self.temp_mem)
        elif sum_reward < 30:
            self.mem30.extend(self.temp_mem)
        elif sum_reward < 40:
            self.mem40.extend(self.temp_mem)
        else:
            self.mem50.extend(self.temp_mem)
        self.temp_mem = []

    def sample_sub_mem(self, mem, num):
        num = min(num, len(mem))
        return random.sample(mem, num), num

    def sample(self, batch_size):
        mem50_sample, remain_50 = self.sample_sub_mem(self.mem50, batch_size // 5)
        remain = batch_size - remain_50
        mem40_sample, remain_40 = self.sample_sub_mem(self.mem40, remain // 4)
        remain = remain - remain_40
        mem30_sample, remain_30 = self.sample_sub_mem(self.mem30, remain // 3)
        remain = remain - remain_30
        mem20_sample, remain_20 = self.sample_sub_mem(self.mem20, remain // 2)
        remain = remain - remain_20
        mem10_sample, remain_10 = self.sample_sub_mem(self.mem10, int(remain * 0.7))
        remain = remain - remain_10
        mem3_sample, remain_3 = self.sample_sub_mem(self.mem3, remain)

        mem_sample = mem3_sample + mem10_sample + mem20_sample + mem30_sample + mem40_sample + mem50_sample

        state, action, reward, next_state, done = zip(*mem_sample)
        state = torch.from_numpy(np.array(state)).float().cuda()
        action = torch.from_numpy(np.array(action)).bool().cuda()
        reward = torch.from_numpy(np.array(reward)).float().cuda()
        next_state = torch.from_numpy(np.array(next_state)).float().cuda()
        done = torch.from_numpy(np.array(done)).int().cuda()
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.mem10)

    def to_disk(self, folder):
        with open(folder + "/mem3", "wb") as f:
            pickle.dump(self.mem3, f)
        with open(folder + "/mem10", "wb") as f:
            pickle.dump(self.mem10, f)
        with open(folder + "/mem20", "wb") as f:
            pickle.dump(self.mem20, f)
        with open(folder + "/mem30", "wb") as f:
            pickle.dump(self.mem30, f)
        with open(folder + "/mem40", "wb") as f:
            pickle.dump(self.mem40, f)
        with open(folder + "/mem50", "wb") as f:
            pickle.dump(self.mem50, f)

    def from_disk(self, folder):
        with open(folder + "/mem3", "rb") as f:
            self.mem3 = pickle.load(f)
        with open(folder + "/mem10", "rb") as f:
            self.mem10 = pickle.load(f)
        with open(folder + "/mem20", "rb") as f:
            self.mem20 = pickle.load(f)
        with open(folder + "/mem30", "rb") as f:
            self.mem30 = pickle.load(f)
        with open(folder + "/mem40", "rb") as f:
            self.mem40 = pickle.load(f)
        with open(folder + "/mem50", "rb") as f:
            self.mem50 = pickle.load(f)


class DQN_Agent:
    def __init__(self, action_size, train=True):
        os.makedirs(f"state_record{os.sep}done", exist_ok=True)
        os.makedirs(f"state_record{os.sep}not_done", exist_ok=True)

        self.action_size = action_size
        self.memory = Memory(action_size)
        # self.memory = Memory(action_size, load_folder="memory")

        self.batch_size = 512
        self.gamma = 0.9

        load_path = "2_244.pth"
        self.model = NeuralNet(load_path).to(DEVICE)
        self.model_fixed = NeuralNet(load_path).to(DEVICE)
        self.writer = SummaryWriter("runs/exp1")

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        if train:
            self.random_action_probability = 1.0
            self.min_random_action_probability = 0.3
            self.random_action_probability_decay = 0.99_999
        else:
            self.random_action_probability = 0.0
            self.min_random_action_probability = 0.0
            self.random_action_probability_decay = 0.0

        self.observe_counter = 0
        self.learn_counter = 0

    def observe_to_disk(self, state, action, reward, next_state, done):
        state = cv2.resize(state, (128, 64))
        next_state = cv2.resize(next_state, (128, 64))
        self.observe_counter += 1
        if done:
            cv2.imwrite(f"state_record{os.sep}done{os.sep}{self.observe_counter}_{action}_{reward}.png", state)
        else:
            image = np.concatenate((state, next_state), axis=0)
            cv2.imwrite(f"state_record{os.sep}not_done{os.sep}{self.observe_counter}_{action}_{reward}.png", image)

    def temp_observe(self, state, action, reward, next_state, done):
        state = cv2.resize(state, (128, 64))
        next_state = cv2.resize(next_state, (128, 64))
        self.memory.temp_save(state, action, reward, next_state, done)
        self.writer.add_scalar("Reward/reward", reward, self.learn_counter)

    def determine_observe(self, sum_reward):
        self.memory.determine_save(sum_reward)

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
        if len(self.memory) < 10_000:
            return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        q_estimate_next_state = self.model_fixed(next_state)
        next_q_value = q_estimate_next_state.gather(1, torch.max(self.model(next_state), 1)[1].unsqueeze(1)).squeeze(1)

        estimate_reward = reward + self.gamma * next_q_value * (1 - done)
        loss = self.loss_function(estimate_reward, self.model(state).masked_select(action))
        loss.backward()
        # print(loss.item())
        self.optimizer.step()

        self.learn_counter += 1

        if self.learn_counter % 10_000 == 0:
            self.memory.to_disk("memory")

            self.model_fixed.load_state_dict(self.model.state_dict())
            print("model_fixed updated")
            print("mem3:", len(self.memory.mem3), "mem10:", len(self.memory.mem10), "mem20:", len(self.memory.mem20),
                  "mem30:", len(self.memory.mem30), "mem40:", len(self.memory.mem40), "mem50:", len(self.memory.mem50))

            self.writer.add_scalar("mem/mem3", len(self.memory.mem3), self.learn_counter)
            self.writer.add_scalar("mem/mem10", len(self.memory.mem10), self.learn_counter)
            self.writer.add_scalar("mem/mem20", len(self.memory.mem20), self.learn_counter)
            self.writer.add_scalar("mem/mem30", len(self.memory.mem30), self.learn_counter)
            self.writer.add_scalar("mem/mem40", len(self.memory.mem40), self.learn_counter)
            self.writer.add_scalar("mem/mem50", len(self.memory.mem50), self.learn_counter)

        if self.learn_counter % 10_000 == 0:
            torch.save(self.model.state_dict(), "pths" + os.sep + f"3_{self.learn_counter // 10_000}" + ".pth")

        self.writer.add_scalar('Loss', loss.item(), self.learn_counter)
        self.writer.add_scalar('Random Action Probability', self.random_action_probability, self.learn_counter)
        self.writer.add_scalar('Reward/q_value', torch.max(self.model(state)).item(), self.learn_counter)
