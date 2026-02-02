# models/encoder.py
import torch
import torch.nn as nn

from models.encoder_cnn import EncoderCNN
from models.bottleneck import BottleneckViT


class Encoder(nn.Module):
    """
    对外暴露给 VQVAE 的 Encoder
    内部结构：CNN → ViT
    输出: (B, h_dim, 64, 64)
    """
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        n_res_layers: int,
        res_h_dim: int,
        vit_layers: int = 2,
        vit_heads: int = 8,
        vit_dim_ff: int | None = None,
        vit_dropout: float = 0.0,
    ):
        super().__init__()

        self.cnn = EncoderCNN(
            in_dim=in_dim,
            h_dim=h_dim,
            n_res_layers=n_res_layers,
            res_h_dim=res_h_dim,
        )

        self.vit = BottleneckViT(
            channels=h_dim,
            num_layers=vit_layers,
            num_heads=vit_heads,
            dim_feedforward=vit_dim_ff,
            dropout=vit_dropout,
            use_residual=True,
        )

    def forward(self, x):
        z = self.cnn(x)   # (B, h_dim, 64, 64)
        z = self.vit(z)   # (B, h_dim, 64, 64)
        return z


if __name__ == "__main__":
    # simple test
    x = torch.randn(2, 1, 256, 256)
    net = Encoder(1, 128, 3, 64)
    z = net(x)
    print("input :", x.shape)
    print("latent:", z.shape)  # (2, 128, 64, 64)
