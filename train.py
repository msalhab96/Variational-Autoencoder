from typing import Union
import os
from pathlib import Path
from config import get_cfg, get_model_args
from data import get_mnist_loaders
from logger import TBLogger
from loss import VAELoss
from model import get_model
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from torch.optim import Adam
from tqdm import tqdm


class Trainer:
    def __init__(
            self,
            train_loader: DataLoader,
            test_loader: DataLoader,
            optimizer: Optimizer,
            criterion: Module,
            model: Module,
            ckpt_dir: Union[Path, str],
            epochs: int,
            logger,
            device
            ) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.logger = logger
        self.epochs = epochs
        self.device = device

    def fit(self):
        for epoch in tqdm(range(self.epochs)):
            self.train()
            self.test()
            self.save_ckpt(epoch)
        print('training completed!')

    def train(self):
        total_loss = 0
        self.model.train()
        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
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
            x = x.to(self.device)
            y = y.to(self.device)
            out, mean, std = self.model(x=x, y=y)
            loss = self.criterion(
                mean=mean, std=std, x=out, y=x
            )
            total_loss += loss.item()
        total_loss /= len(self.test_loader)
        self.logger.log_mean(mean[0].mean().item())
        self.logger.log_std(std[0].mean().item())
        self.logger.log_test_loss(total_loss)


def get_optim(model):
    return Adam(model.parameters())


def get_trainer(cfg):
    model = get_model(cfg, get_model_args(cfg))
    model = model.to(cfg.device)
    train_loader, test_loader = get_mnist_loaders(
        cfg.data_path, cfg.batch_size
        )
    optim = get_optim(model)
    criterion = VAELoss(beta=cfg.beta).to(cfg.device)
    logger = TBLogger(cfg.logdir)
    return Trainer(
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optim,
        criterion=criterion,
        model=model,
        ckpt_dir=cfg.ckpt_dir,
        epochs=cfg.epochs,
        logger=logger,
        device=cfg.device
    )


if __name__ == '__main__':
    cfg = get_cfg()
    get_trainer(cfg).fit()
