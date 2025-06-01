import torch
import torch.nn as nn
from models.patch_embedding import PatchEmbedding
from models.residual import ResidualStack
import numpy as np

class Encoder(nn.Module):
    def __init__(self, in_channels=1, emb_dim=128, patch_size=16, img_size=256,
                 num_layers=6, num_heads=8, use_residual=True,
                 n_res_layers=3, res_h_dim=64):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, emb_dim, patch_size, img_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=num_heads,
            dim_feedforward=emb_dim * 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.emb_dim = emb_dim
        self.grid_size = img_size // patch_size

        self.use_residual = use_residual
        if self.use_residual:
            self.res_stack = ResidualStack(emb_dim, emb_dim, res_h_dim, n_res_layers)

    def forward(self, x):
        x, (H, W) = self.patch_embed(x)
        x = self.transformer(x)
        x = x.transpose(1, 2).view(-1, self.emb_dim, H, W)

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
