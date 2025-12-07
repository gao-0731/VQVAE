import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualStack


class LayerNorm2d(nn.Module):
    """在通道维上做 LayerNorm 的 2D 版本"""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias


class MHSA2D(nn.Module):
    """
    全局多头自注意力（空间位置当 token）。
    输入/输出: [B, C_in, H, W]，保持通道数不变。
    """
    def __init__(self, in_channels, attn_dim, num_heads=8, qkv_bias=True,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        assert attn_dim % num_heads == 0
        self.in_channels = in_channels
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 1x1 降维到 attn_dim 后做 qkv
        self.pre = nn.Conv2d(in_channels, attn_dim, kernel_size=1, bias=True)
        self.norm = LayerNorm2d(attn_dim)
        self.qkv = nn.Conv2d(attn_dim, attn_dim * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # 回投到 in_channels，保持与上游一致
        self.proj = nn.Conv2d(attn_dim, in_channels, kernel_size=1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        x = self.pre(x)                 # [B, attn_dim, H, W]
        x = self.norm(x)
        qkv = self.qkv(x)               # [B, 3*attn_dim, H, W]
        q, k, v = torch.chunk(qkv, 3, dim=1)

        def reshape_heads(t):
            # [B, attn_dim, H, W] -> [B, heads, HW, head_dim]
            B_, C_, H_, W_ = t.shape
            t = t.view(B_, self.num_heads, self.head_dim, H_*W_)
            return t.permute(0, 1, 3, 2).contiguous()

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale     # [B, heads, HW, HW]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v                                    # [B, heads, HW, head_dim]
        out = out.permute(0, 1, 3, 2).contiguous()        # [B, heads, head_dim, HW]
        out = out.view(B, self.attn_dim, H, W)            # [B, attn_dim, H, W]

        out = self.proj(out)                              # [B, C, H, W]
        out = self.proj_drop(out)
        return residual + out


class WindowMHSA2D(nn.Module):
    """
    窗口注意力（不重叠窗口），复杂度显著降低。
    输入/输出: [B, C_in, H, W]，保持通道数不变。
    需要 H、W 能被 window_size 整除（或在外面 padding）。
    """
    def __init__(self, in_channels, attn_dim, num_heads=8, window_size=8,
                 qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert attn_dim % num_heads == 0
        self.in_channels = in_channels
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.pre = nn.Conv2d(in_channels, attn_dim, kernel_size=1, bias=True)
        self.norm = LayerNorm2d(attn_dim)
        self.qkv = nn.Conv2d(attn_dim, attn_dim * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(attn_dim, in_channels, kernel_size=1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        M = self.window_size
        assert H % M == 0 and W % M == 0, \
            f"H/W 必须能被 window_size={M} 整除，当前 H={H}, W={W}"

        x = self.pre(x)
        x = self.norm(x)
        qkv = self.qkv(x)  # [B, 3*attn_dim, H, W]
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # 切成窗口：[B, attn_dim, H, W] -> [Bn, attn_dim, M, M]
        def window_partition(t):
            B_, C_, H_, W_ = t.shape
            t = t.view(B_, C_, H_//M, M, W_//M, M) \
                 .permute(0, 2, 4, 1, 3, 5).contiguous()
            return t.view(-1, C_, M, M)  # Bn = B * (H/M) * (W/M)

        q = window_partition(q)
        k = window_partition(k)
        v = window_partition(v)

        HWw = M * M

        def reshape_heads(t):
            # [Bn, attn_dim, M, M] -> [Bn, heads, HWw, head_dim]
            Bn, Cn, Mh, Mw = t.shape
            t = t.view(Bn, self.num_heads, self.head_dim, HWw)
            return t.permute(0, 1, 3, 2).contiguous()

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale       # [Bn, heads, HWw, HWw]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v                                      # [Bn, heads, HWw, head_dim]
        out = out.permute(0, 1, 3, 2).contiguous()          # [Bn, heads, head_dim, HWw]
        out = out.view(out.size(0), self.attn_dim, M, M)    # [Bn, attn_dim, M, M]

        # 合并窗口回原图
        Bn = out.size(0)
        num_h = H // M
        num_w = W // M
        out = out.view(B, num_h, num_w, self.attn_dim, M, M) \
                 .permute(0, 3, 1, 4, 2, 5).contiguous().view(B, self.attn_dim, H, W)

        out = self.proj(out)
        out = self.proj_drop(out)
        return residual + out


class Encoder(nn.Module):
    """
    你的原始 Encoder + 注意力块。
    conv_downsample(×2) -> conv(stride=1) -> ResidualStack -> Attention
    """
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim,
                 attn_type="window", attn_dim_ratio=0.5, num_heads=8,
                 window_size=8, attn_drop=0., proj_drop=0.):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        )

        attn_dim = max(32, int(h_dim * attn_dim_ratio))
        if attn_type == "global":
            self.attn = MHSA2D(h_dim, attn_dim, num_heads=num_heads,
                               attn_drop=attn_drop, proj_drop=proj_drop)
        elif attn_type == "window":
            self.attn = WindowMHSA2D(h_dim, attn_dim, num_heads=num_heads,
                                     window_size=window_size,
                                     attn_drop=attn_drop, proj_drop=proj_drop)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

    def forward(self, x):
        x = self.conv_stack(x)   # [B, h_dim, H/4, W/4]
        x = self.attn(x)         # 形状不变
        return x


if __name__ == "__main__":
    # 正确的 PyTorch 形状是 [B, C, H, W]。这里举例 B=3, C=40, H=W=200
    B, C, H, W = 3, 40, 200, 200
    x = torch.randn(B, C, H, W)

    # 用你的参数测试（通道=40）
    encoder = Encoder(in_dim=C, h_dim=128, n_res_layers=3, res_h_dim=64,
                      attn_type="window",  # 或 "global"
                      attn_dim_ratio=0.5, num_heads=8, window_size=8)

    y = encoder(x)
    print("Input  :", x.shape)
    print("Output :", y.shape)  # -> [B, 128, H/4, W/4] 这里是 [3, 128, 50, 50]
