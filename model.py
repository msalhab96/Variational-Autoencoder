import torch
import torch.nn as nn
from layers import FCEncoder, FCDecoder
from torch import Tensor
from utils import parameterize


class FFModel(nn.Module):
    def __init__(
            self,
            in_size: int,
            hidden_size: int,
            latent_size: int,
            n_layers: int,
            p_dropout: float,
            ) -> None:
        super().__init__()
        self.encoder = FCEncoder(
            in_size=in_size,
            hidden_size=hidden_size,
            latent_size=hidden_size,
            n_layers=n_layers,
            p_dropout=p_dropout
        )
        self.decoder = FCDecoder(
            out_size=in_size,
            latent_size=hidden_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            p_dropout=p_dropout
        )
        self.fc_mu = nn.Linear(
            in_features=hidden_size,
            out_features=latent_size
        )
        self.fc_std = nn.Linear(
            in_features=hidden_size,
            out_features=latent_size
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = x.view(x.shape[0], -1)
        out = self.encoder(x)
        mean = self.fc_mu(out)
        std = self.fc_std(out)
        z = parameterize(mean, std)
        out = self.decoder(z)
        return out, mean, std


class CFFModel(nn.Module):
    def __init__(
            self,
            n_class: int,
            cond_size: int,
            in_size: int,
            hidden_size: int,
            latent_size: int,
            n_layers: int,
            p_dropout: float,
            ) -> None:
        super().__init__()
        self.emb = nn.Embedding(
            num_embeddings=n_class,
            embedding_dim=cond_size
        )
        self.encoder = FCEncoder(
            in_size=in_size + cond_size,
            hidden_size=hidden_size,
            latent_size=hidden_size,
            n_layers=n_layers,
            p_dropout=p_dropout
        )
        self.decoder = FCDecoder(
            out_size=in_size,
            latent_size=hidden_size + cond_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            p_dropout=p_dropout
        )
        self.fc_mu = nn.Linear(
            in_features=hidden_size,
            out_features=latent_size
        )
        self.fc_std = nn.Linear(
            in_features=hidden_size,
            out_features=latent_size
        )

    def forward(self, x: Tensor, y: Tensor, *args, **kwargs) -> Tensor:
        x = x.view(x.shape[0], -1)
        cond = self.emb(y)
        x = torch.cat([x, cond], dim=1)
        out = self.encoder(x)
        mean = self.fc_mu(out)
        std = self.fc_std(out)
        z = parameterize(mean, std)
        z = torch.cat([z, cond], dim=1)
        out = self.decoder(z)
        return out, mean, std
