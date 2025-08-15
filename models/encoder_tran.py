import torch
import torch.nn as nn
from models.patch_embedding import PatchEmbedding
from models.residual import ResidualStack
import numpy as np

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels=1, emb_dims=[128, 64, 32], patch_size=16, img_size=256,
                 num_layers=6, num_heads=32, use_residual=True, n_res_layers=3, res_h_dim=64):
        super().__init__()
        # emb_dims: [128, 64, 32] のように段階的に減らす

        self.patch_embed = PatchEmbedding(in_channels, emb_dims[0], patch_size, img_size)
        self.transformer_blocks = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.num_stages = len(emb_dims)
        self.emb_dims = emb_dims

        # n_layers_per_stage = num_layers // num_stages
        n_layers_per_stage = num_layers // (self.num_stages)
        for i in range(self.num_stages):
            d_model = emb_dims[i]
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads,
                dim_feedforward=d_model * 4, batch_first=True)
            # 複数層まとめて
            block = nn.TransformerEncoder(encoder_layer, num_layers=n_layers_per_stage)
            self.transformer_blocks.append(block)

            # 最終段以外は次元圧縮線形層
            if i < self.num_stages - 1:
                self.linear_layers.append(nn.Linear(d_model, emb_dims[i+1]))

        self.grid_size = img_size // patch_size
        self.use_residual = use_residual
        if use_residual:
            self.res_stack = ResidualStack(emb_dims[-1], emb_dims[-1], res_h_dim, n_res_layers)

    def forward(self, x):
        x, (H, W) = self.patch_embed(x)   # [B, HW, D0]
        for i in range(self.num_stages):
            x = self.transformer_blocks[i](x)
            if i < self.num_stages - 1:
                x = self.linear_layers[i](x)  # 次元圧縮
        # [B, HW, D_last]
        x = x.transpose(1, 2).view(-1, self.emb_dims[-1], H, W)
        if self.use_residual:
            x = self.res_stack(x)
        return x


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(40, 128, 3, 64)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
