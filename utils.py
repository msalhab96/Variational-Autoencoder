import torch


def parameterize(meu, std):
    eps = torch.randn(*std.shape).to(meu.device)
    return meu + torch.exp(std) * eps
