import math
import torch
from torch import nn
from torch import Tensor


class ConvBlock(nn.Module):
    def __init__(
            self,
            kernel_size: int,
            in_channels: int,
            out_channels: int,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size
            )
        self.b_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.b_norm(out)
        out = self.dropout(out)
        return out


class TransConvBlock(ConvBlock):
    def __init__(
            self,
            kernel_size: int,
            in_channels: int,
            out_channels: int,
            p_dropout: float
            ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            p_dropout=p_dropout
        )
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size
            )

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class FCBlock(nn.Module):
    def __init__(
            self,
            in_size: int,
            out_size: int,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.fc = nn.Linear(
            in_features=in_size, out_features=out_size
            )
        self.b_norm = nn.BatchNorm1d(out_size)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc(x)
        out = self.b_norm(out)
        out = self.dropout(out)
        return out


class ConvEncoder(nn.Module):
    def __init__(
            self,
            n_layers: int,
            kernel_size: int,
            in_channels: int,
            out_channels: int,
            img_size: int,
            latent_size: int,
            h_size: int,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            ConvBlock(
                in_channels=in_channels if i == 1 else (i - 1) * out_channels,
                out_channels=i * out_channels if i != n_layers else 1,
                kernel_size=kernel_size,
                p_dropout=p_dropout
            )
            for i in range(1, 1 + n_layers)
        ])
        size = img_size - n_layers * (kernel_size - 1)
        self.fc1 = nn.Linear(
            in_features=size ** 2, out_features=h_size
            )
        self.fc2 = nn.Linear(
            in_features=h_size, out_features=latent_size
            )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        out = x.view(x.shape[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class CondConvEncoder(ConvEncoder):
    def __init__(
            self,
            n_layers: int,
            kernel_size: int,
            in_channels: int,
            out_channels: int,
            img_size: int,
            latent_size: int,
            h_size: int,
            p_dropout: float,
            n_classes: int
            ) -> None:
        super().__init__(
            n_layers=n_layers,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            latent_size=latent_size,
            h_size=h_size,
            p_dropout=p_dropout
        )
        self.emb = nn.Embedding(
            num_embeddings=n_classes,
            embedding_dim=img_size
        )
        self.img_size = img_size

    def add_cond(self, x: Tensor, labels: Tensor) -> Tensor:
        # x of shape [B, C, H, W]
        # labels of shape [B]
        (b, c, h, w) = x.shape
        cond_img = torch.zeros(b, c, h + 1, w + 1).to(x.device)
        cond = self.emb(labels)
        cond = cond.unsqueeze(1)
        cond_img[..., -1, :] = cond_img[..., -1, :] + cond
        cond_img[..., :-1, :-1] = cond_img[..., :-1, :-1] + x
        return cond_img

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        x = self.add_cond(x, labels)
        return super().forward(x)


class ConvDecoder(nn.Module):
    def __init__(
            self,
            n_layers: int,
            kernel_size: int,
            out_channels: int,
            latent_size: int,
            h_size: int,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransConvBlock(
                in_channels=1 if i == 1 else (i - 1) * out_channels,
                out_channels=i * out_channels if i != n_layers else 1,
                kernel_size=kernel_size,
                p_dropout=p_dropout
            )
            for i in range(1, 1 + n_layers)
        ])
        self.fc1 = nn.Linear(
            in_features=latent_size, out_features=h_size
            )
        self.fc2 = nn.Linear(
            in_features=h_size, out_features=latent_size
            )
        self.latent_size = latent_size

    def forward(self, z: Tensor) -> Tensor:
        # z is the parmterized tensor of shape [B, l]
        z = self.fc1(z)
        z = self.fc2(z)
        size = int(math.sqrt(self.latent_size))
        z = z.view(z.shape[0], 1, size, size)
        out = z
        for layer in self.layers:
            out = layer(out)
        return out


class FCEncoder(nn.Module):
    def __init__(
            self,
            in_size: int,
            hidden_size: int,
            latent_size: int,
            n_layers: int,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            FCBlock(
                in_size=in_size if i == 1 else hidden_size // (i - 1),
                out_size=latent_size if i == n_layers else hidden_size // i,
                p_dropout=p_dropout
            )
            for i in range(1, 1 + n_layers)
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class FCDecoder(nn.Module):
    def __init__(
            self,
            out_size: int,
            latent_size: int,
            hidden_size: int,
            n_layers: int,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            FCBlock(
                in_size=latent_size if i == 1 else hidden_size * (i - 1),
                out_size=out_size if i == n_layers else hidden_size * i,
                p_dropout=p_dropout
            )
            for i in range(1, 1 + n_layers)
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
