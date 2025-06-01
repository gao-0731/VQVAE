# models/patch_embedding.py
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, emb_dim=128, patch_size=16, img_size=256):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))

    def forward(self, x):
        x = self.proj(x)  # [B, emb_dim, H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, emb_dim]
        x = x + self.pos_embedding
        return x, (H, W)  # return spatial for later reshape
