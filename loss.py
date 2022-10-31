import torch
from torch import nn
from torch import Tensor


class KLDiv(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, mean: Tensor, std: Tensor):
        mu_squared = mean ** 2
        std = torch.log(std)
        std_squred = std ** 2
        loss = 1 + std - mu_squared - std_squred
        loss = loss.sum(dim=-1)
        return -0.5 * loss


class MSE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(x: Tensor, y: Tensor) -> Tensor:
        loss = (x - y) ** 2
        loss = loss.sum(dim=-1)
        return loss


class VAELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kl = KLDiv()
        self.mse = MSE()

    def forward(
            self, mean: Tensor, std: Tensor, x: Tensor, y: Tensor
            ) -> Tensor:
        loss = self.kl(mean, std) + self.mse(x, y)
        return loss.mean()
