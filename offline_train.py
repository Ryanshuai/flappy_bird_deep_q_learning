import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

from dataset import QvalueDataset
from dqn_agent import NeuralNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_len = 50_000

epoch = 10
start_sample_rate = 0.3
end_sample_rate = 0.1

gamma = 0.99

dataset = QvalueDataset("state_record", done_sample_rate=start_sample_rate, dataset_len=dataset_len)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
writer = SummaryWriter("runs/exp1")

model = NeuralNet("pths/model4_9.pth").to(DEVICE)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch_idx in range(epoch):
    sample_rate = start_sample_rate - (start_sample_rate - end_sample_rate) * epoch_idx / epoch

    for batch_idx, data in enumerate(dataloader):
        print(f"epoch: {epoch_idx}, batch: {batch_idx}, sample_rate: {sample_rate}")

        state, next_state, action, reward, done = data
        state = state.float().to(DEVICE)
        next_state = next_state.float().to(DEVICE)
        action = action.to(DEVICE)
        reward = reward.float().to(DEVICE)
        done = done.to(DEVICE)

        q_estimate_next_state = model(next_state)
        max_q_estimate_next_state = torch.max(q_estimate_next_state, dim=1)[0]
        estimate_reward = reward + gamma * max_q_estimate_next_state * (1 - done)
        loss = loss_function(estimate_reward, model(state).masked_select(action))
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss", loss.item(), epoch_idx * len(dataloader) + batch_idx)
        writer.add_scalar("Sample Rate", sample_rate, epoch_idx * len(dataloader) + batch_idx)

    torch.save(model.state_dict(), f"pths{os.sep}model5_{epoch_idx}.pth")
