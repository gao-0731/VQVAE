import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, emb_dim=128, patch_size=16, img_size=256, dropout=0.0):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.emb_dim = emb_dim

        # パッチ分割 & 埋め込み
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

        # sinusoidal position embedding を事前計算
        self.register_buffer("pos_embedding", self._build_sinusoidal_embedding(self.num_patches, emb_dim))
        self.dropout = nn.Dropout(dropout)

    def _build_sinusoidal_embedding(self, num_positions, dim):
        """
        標準的な Transformer の Sinusoidal 位置埋め込みを生成
        [num_positions, dim]
        """
        pe = torch.zeros(num_positions, dim)
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)  # [num_positions, 1]
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, num_positions, dim]

    def forward(self, x):
        x = self.proj(x)  # [B, emb_dim, H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, emb_dim]
        x = x + self.pos_embedding[:, :x.size(1), :].to(x.device)  # broadcasting
        x = self.dropout(x)
        return x, (H, W)  # return spatial for later reshape
