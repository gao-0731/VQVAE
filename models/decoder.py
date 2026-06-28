import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()

        self.inverse_conv_stack = nn.Sequential(

            # 64x64
            nn.ConvTranspose2d(
                in_dim,
                h_dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(inplace=True),

            # 64 -> 128
            nn.ConvTranspose2d(
                h_dim,
                h_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(inplace=True),

            # 128 -> 256
            nn.ConvTranspose2d(
                h_dim // 2,
                1,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)