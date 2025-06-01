# models/decoder.py
import torch
import torch.nn as nn
from models.residual import ResidualStack

class Decoder(nn.Module):
    def __init__(self, out_channels=1, emb_dim=128, patch_size=16, img_size=256,
                 num_layers=6, num_heads=8, use_residual=True,
                 n_res_layers=3, res_h_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.emb_dim = emb_dim
        self.out_channels = out_channels
        self.use_residual = use_residual

        self.pos_embedding = nn.Parameter(torch.randn(1, self.grid_size ** 2, emb_dim))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=num_heads,
            dim_feedforward=emb_dim * 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        if use_residual:
            self.res_stack = ResidualStack(emb_dim, emb_dim, res_h_dim, n_res_layers)

        self.out_proj = nn.Sequential(
            nn.Linear(emb_dim, patch_size * patch_size * out_channels),
            nn.ReLU(),  # 对齐 CNN decoder 的中间 ReLU
        )

    def forward(self, x):
        # 输入: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = x + self.pos_embedding
        x = self.transformer(x)

        # ResidualStack (与 CNN 相同结构位置)
        if self.use_residual:
            x_reshaped = x.transpose(1, 2).view(B, self.emb_dim, self.grid_size, self.grid_size)
            x_reshaped = self.res_stack(x_reshaped)
            x = x_reshaped.flatten(2).transpose(1, 2)  # 回到 [B, N, C]

        # 输出图像还原
        x = self.out_proj(x)  # [B, N, patch * patch * C_out]
        x = x.view(B, self.grid_size, self.grid_size, self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.out_channels, H * self.patch_size, W * self.patch_size)
        return x
