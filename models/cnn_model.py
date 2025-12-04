"""Neural models for pose-based anomaly detection."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseAutoencoder(nn.Module):
    """Temporal-spatial convolutional autoencoder for pose sequences."""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_dims: tuple[int, ...] = (32, 64, 96),
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        encoder_layers = []
        c_in = in_channels
        for idx, c_out in enumerate(hidden_dims):
            stride = 2 if idx < len(hidden_dims) - 1 else 1
            encoder_layers.extend(
                [
                    nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                ]
            )
            c_in = c_out
        self.encoder = nn.Sequential(*encoder_layers)
        self.latent_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_in, latent_dim),
        )
        self.decoder_head = nn.Sequential(nn.Linear(latent_dim, c_in), nn.ReLU(inplace=True))

        decoder_layers = []
        c_dec = c_in
        for c_out in reversed(hidden_dims[:-1]):
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(c_dec, c_out, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                ]
            )
            c_dec = c_out
        decoder_layers.append(nn.Conv2d(c_dec, in_channels, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor):
        """Return reconstruction and latent vector."""

        encoded = self.encoder(x)
        latent = self.latent_head(encoded)
        seed = self.decoder_head(latent).view(x.size(0), -1, 1, 1)
        recon = self.decoder(seed)
        recon = F.interpolate(recon, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        return recon, latent
