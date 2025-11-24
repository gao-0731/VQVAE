# models/bottleneck_vit.py
import torch
import torch.nn as nn


class BottleneckViT(nn.Module):
    """
    在 CNN 输出特征图上执行 Transformer
    输入:  (B, C, H, W) 例如 (B, 128, 64, 64)
    输出:  (B, C, H, W)
    """
    def __init__(
        self,
        channels: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        use_residual: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.use_residual = use_residual

        if dim_feedforward is None:
            dim_feedforward = channels * 4

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # lazy 初始化
        self.pos_embed: nn.Parameter | None = None

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W

        tokens = x.flatten(2).transpose(1, 2)   # (B, L, C)

        if (self.pos_embed is None) or (self.pos_embed.size(1) != L):
            pe = torch.zeros(1, L, C, device=x.device)
            nn.init.trunc_normal_(pe, std=0.02)
            self.pos_embed = nn.Parameter(pe)

        tokens_with_pos = tokens + self.pos_embed

        out = self.transformer(tokens_with_pos)

        if self.use_residual:
            out = out + tokens

        out = out.transpose(1, 2).view(B, C, H, W)
        return out
