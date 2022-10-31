from typing import Union
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class TBLogger(SummaryWriter):
    def __init__(self, log_dir: Union[str, Path], *args, **kwargs) -> None:
        super().__init__(log_dir=log_dir, *args, **kwargs)
        self.mean_counter = 0
        self.std_counter = 0
        self.train_loss_counter = 0
        self.test_loss_counter = 0

    def log_mean(self, value):
        self.add_scalar(
            'mean', value, self.mean_counter
            )
        self.mean_counter += 1

    def log_std(self, value):
        self.add_scalar(
            'std', value, self.std_counter
            )
        self.std_counter += 1

    def log_train_loss(self, value):
        self.add_scalar(
            'train_loss', value, self.train_counter
            )
        self.train_counter += 1

    def log_test_loss(self, value):
        self.add_scalar(
            'test_loss', value, self.test_counter
            )
        self.test_counter += 1
