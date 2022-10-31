from typing import Union
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from interfaces import ILogger


class Trainer:
    def __init__(
            self,
            train_loader: DataLoader,
            test_loader: DataLoader,
            optimizer: Optimizer,
            criterion: Module,
            model: Module,
            ckpt_dir: Union[Path, str],
            logger: ILogger,
            epochs: int
            ) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.logger = logger
        self.epochs = epochs

    def fit(self):
        for i in range(self.epochs):
            self.train()
            self.test()
            self.save_ckpt()
        print('training completed!')

    def train(self):
        total_loss = 0
        self.model.train()
        for x, y in self.train_loader:
            self.optimizer.zero_grad()
            out, mean, std = self.model(x=x, y=y)
            loss = self.criterion(
                mean=mean, std=std, x=out, y=x
            )
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        total_loss /= len(self.train_loader)
        self.logger.log_train_loss(total_loss)

    def save_ckpt(self, epoch: int):
        path = os.path.join(self.ckpt_dir, f'checlpoint_{epoch}')
        torch.save(self.model.state_dict(), path)

    def test(self):
        total_loss = 0
        self.model.eval()
        for x, y in self.test_loader:
            out, mean, std = self.model(x=x, y=y)
            loss = self.criterion(
                mean=mean, std=std, x=out, y=x
            )
            total_loss += loss.item()
        total_loss /= len(self.test_loader)
        self.logger.log_mean(mean[0].mean().item())
        self.logger.log_std(std[0].mean().item())
        self.logger.log_test_loss(total_loss)
