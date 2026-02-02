# models/encoder_cnn.py
import torch
import torch.nn as nn
from models.residual import ResidualStack


class EncoderCNN(nn.Module):
    """
    纯 CNN 编码器
    输入:  (B, in_dim, 256, 256)
    输出:  (B, h_dim, 64, 64)
    """
    def __init__(self, in_dim: int, h_dim: int, n_res_layers: int, res_h_dim: int):
        super().__init__()
        kernel = 4
        stride = 2

        self.conv_stack = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(
                in_channels=in_dim,
                out_channels=h_dim // 2,
                kernel_size=kernel,
                stride=stride,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            # 128 -> 64
            nn.Conv2d(
                in_channels=h_dim // 2,
                out_channels=h_dim,
                kernel_size=kernel,
                stride=stride,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            # 保持 64x64
            nn.Conv2d(
                in_channels=h_dim,
                out_channels=h_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            ResidualStack(
                in_dim=h_dim,
                h_dim=h_dim,
                res_h_dim=res_h_dim,
                n_res_layers=n_res_layers,
            ),
        )

    def forward(self, x):
        return self.conv_stack(x)
