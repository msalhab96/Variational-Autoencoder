from pathlib import Path
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Union


def get_mnist_loaders(path: Union[Path, str], batch_size: int):
    train_data = MNIST(
        path,
        transform=transforms.Compose([transforms.ToTensor(), ]),
        train=True,
        download=True
        )
    test_dataset = MNIST(
        path,
        transform=transforms.Compose([transforms.ToTensor(), ]),
        train=False,
        download=True
        )
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True
        )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True
        )
    return train_loader, test_loader
