import torch
from torch import nn
from torch import Tensor


class KLDiv(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, mean: Tensor, std: Tensor):
        mu_squared = mean ** 2
        loss = 1 + std - mu_squared - torch.exp(std)
        return -0.5 * loss.sum(dim=-1)


class MSE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        loss = (x - y) ** 2
        return loss.sum(dim=-1)


class VAELoss(nn.Module):
    def __init__(self, beta=1.0) -> None:
        super().__init__()
        self.kl = KLDiv()
        self.mse = MSE()
        self.beta = beta

    def forward(
            self, mean: Tensor, std: Tensor, x: Tensor, y: Tensor
            ) -> Tensor:
        mse = self.mse(x, y.view(y.shape[0], -1))
        kl = self.kl(mean, std)
        loss = self.beta * kl + mse
        return loss.mean()
