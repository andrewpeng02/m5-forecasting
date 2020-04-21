import os
import click

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.m5_dataset import M5Dataset
from src.lstm_model import M5LSTM


@click.command()
@click.argument('lr', type=float,  default=0.001)
@click.argument('num_epochs', type=int,  default=20)
@click.argument('batch_size', type=int,  default=32)
@click.argument('seq_length', type=int,  default=365)
@click.argument('hidden_dim', type=int,  default=128)
@click.argument('num_layers', type=int,  default=2)
def main(**kwargs):
    project_dir = os.getcwd()[:-3]

    train_dataset = M5Dataset(project_dir + '/data/out/train.npy', kwargs['seq_length'])
    train_loader = DataLoader(train_dataset, batch_size=kwargs['batch_size'], shuffle=True, num_workers=4)

    model = M5LSTM(4, kwargs['hidden_dim'], kwargs['num_layers'])

    optimizer = optim.Adam(model.parameters(), lr=kwargs['lr'])
    losses = train(train_loader, model, optimizer, kwargs['batch_size'], kwargs['num_epochs'], project_dir)

    np.save(project_dir + '/notebooks/losses.npy', losses)


def train(train_loader, model, optimizer, batch_size, num_epochs, project_dir, device='cuda'):
    model = model.to(device)
    losses = []

    for epoch in range(num_epochs):
        total_loss = 0

        for i, (seqs, labels) in enumerate(train_loader):
            if seqs.shape[0] != batch_size:
                break

            hidden = model.init_hidden(batch_size)
            seqs, labels = seqs.float().to(device), labels.float().to(device)

            optimizer.zero_grad()
            out, hidden = model(seqs, hidden)

            # RMSE loss
            loss = torch.sqrt(F.mse_loss(out, labels.flatten().unsqueeze(1)))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        torch.save(model, project_dir + '/models/checkpoint.pth')
        print(f'Epoch [{epoch + 1} / {num_epochs}] \t Loss {total_loss / len(train_loader)}')
        losses.append(total_loss / len(train_loader))
    return losses


if __name__ == "__main__":
    main()
