import torch
import torch.nn as nn


# ============ Multi-Head Self-Attention ============
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim: int, head: int = 8, dropout: float = 0.):
        super().__init__()
        assert emb_dim % head == 0, "emb_dim 必须能被 head 整除"
        self.head = head
        self.d_k = emb_dim // head

        self.qkv = nn.Linear(emb_dim, emb_dim * 3)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.head, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, head, N, d_k]

        attn = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # [B, head, N, d_k]
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        return out


# ============ ViT Block (Pre-LN) ============
class ViTBlock(nn.Module):
    def __init__(self, emb_dim=384, head=8, mlp_ratio=4, dropout=0.):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.msa = MultiHeadSelfAttention(emb_dim, head, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * mlp_ratio, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.msa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ============ Patch Embedding ============
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, emb_dim=384, patch_size=16, image_size=256):
        super().__init__()
        assert image_size % patch_size == 0, "image_size 必须能被 patch_size 整除"
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_channels, emb_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, D, H', W']
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x, (H, W)


# ============ ViT Encoder ============
class ViTEncoder(nn.Module):
    def __init__(self, in_channels=1, emb_dim=128,
                 patch_size=16, image_size=256,
                 num_blocks=8, head=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, emb_dim, patch_size, image_size)
        self.blocks = nn.Sequential(*[
            ViTBlock(emb_dim, head, mlp_ratio, dropout) for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.emb_dim = emb_dim

    def forward(self, x):
        # patch embedding
        x, (H, W) = self.patch_embed(x)  # [B, N, D]

        # transformer blocks
        x = self.blocks(x)
        x = self.norm(x)  # [B, N, D]

        # reshape to feature map [B, D, H, W]
        B = x.size(0)
        x = x.transpose(1, 2).contiguous().view(B, self.emb_dim, H, W)
        return x


# ============ 测试 ============
if __name__ == "__main__":
    img = torch.randn(2, 1, 256, 256)  # [B, C, H, W]
    model = ViTEncoder(in_channels=1, emb_dim=128,
                       patch_size=16, image_size=256,
                       num_blocks=8, head=8)
    out = model(img)
    print("输出 shape:", out.shape)  # [2, 128, 16, 16]
