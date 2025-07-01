# # models/decoder.py
# import torch
# import torch.nn as nn
from models.residual import ResidualStack

# class Decoder(nn.Module):
#     def __init__(self, out_channels=1, emb_dim=128, patch_size=16, img_size=256,
#                  num_layers=6, num_heads=8, use_residual=True,
#                  n_res_layers=3, res_h_dim=64):
#         super().__init__()
#         self.patch_size = patch_size
#         self.grid_size = img_size // patch_size
#         self.emb_dim = emb_dim
#         self.out_channels = out_channels
#         self.use_residual = use_residual

#         self.pos_embedding = nn.Parameter(torch.randn(1, self.grid_size ** 2, emb_dim))

#         decoder_layer = nn.TransformerEncoderLayer(
#             d_model=emb_dim, nhead=num_heads,
#             dim_feedforward=emb_dim * 4, batch_first=True)
#         self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

#         if use_residual:
#             self.res_stack = ResidualStack(emb_dim, emb_dim, res_h_dim, n_res_layers)

#         self.out_proj = nn.Sequential(
#             nn.Linear(emb_dim, patch_size * patch_size * out_channels),
#             nn.ReLU(),  # 对齐 CNN decoder 的中间 ReLU
#         )

#     def forward(self, x):
#         # 输入: [B, C, H, W]
#         B, C, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)  # [B, N, C]
#         x = x + self.pos_embedding
#         x = self.transformer(x)

#         # ResidualStack (与 CNN 相同结构位置)
#         if self.use_residual:
#             x_reshaped = x.transpose(1, 2).view(B, self.emb_dim, self.grid_size, self.grid_size)
#             x_reshaped = self.res_stack(x_reshaped)
#             x = x_reshaped.flatten(2).transpose(1, 2)  # 回到 [B, N, C]

#         # 输出图像还原
#         x = self.out_proj(x)  # [B, N, patch * patch * C_out]
#         x = x.view(B, self.grid_size, self.grid_size, self.patch_size, self.patch_size, self.out_channels)
#         x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
#         x = x.view(B, self.out_channels, H * self.patch_size, W * self.patch_size)
#         return x


import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, out_channels=1, emb_dims=[128, 256, 512, 256], num_layers=6, num_heads=8):
        super().__init__()
        self.emb_dims = emb_dims

        # 通道升维
        self.initial_linear = nn.Linear(emb_dims[0], emb_dims[1])  # 128->256

        # transformer blocks
        self.transformer_block = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_dims[1], nhead=num_heads, dim_feedforward=emb_dims[1]*4, batch_first=True),
            num_layers=num_layers
        )

        # reshape回feature map
        # 假设输入64x64，输出也是64x64
        self.feature_reshape_dim = 64

        # 上采样：256->512->256->1
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),       # 64x64 -> 128x128
            nn.Conv2d(emb_dims[1], emb_dims[2], 3, padding=1), # 256->512
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),       # 128x128 -> 256x256
            nn.Conv2d(emb_dims[2], emb_dims[3], 3, padding=1), # 512->256
            nn.ReLU(),
            nn.Conv2d(emb_dims[3], out_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z_q):
        B, C, H, W = z_q.shape  # [B, 128, 64, 64]
        x = z_q.permute(0,2,3,1).reshape(B, H*W, C)    # [B, 4096, 128]
        x = self.initial_linear(x)                      # [B, 4096, 256]
        x = self.transformer_block(x)                   # [B, 4096, 256]
        x = x.view(B, H, W, self.emb_dims[1]).permute(0, 3, 1, 2).contiguous()  # [B, 256, 64, 64]
        x = self.upsample(x)                            # [B, 1, 256, 256]
        return x
