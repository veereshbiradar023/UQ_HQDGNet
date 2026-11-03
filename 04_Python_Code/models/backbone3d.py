"""
3D Classical Hybrid Backbone:
3D Residual Convolutional Blocks with 3D Spatial-Contextual Self-Attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    """
    3D Residual Block with GroupNorm and GELU non-linearities.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.act(out + res)

class SpatialAttention3D(nn.Module):
    """
    Lightweight 3D Multi-Head Self-Attention over downsampled spatial tokens.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        N = D * H * W
        tokens = x.permute(0, 2, 3, 4, 1).reshape(B, N, C)  # [B, N, C]
        norm_tokens = self.norm(tokens)

        qkv = self.qkv(norm_tokens).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = tokens + out
        out = out.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        return out

class ClassicalHybridBackbone3D(nn.Module):
    """
    Multi-Scale 3D Feature Extractor.
    Input: [B, 1, D, H, W]
    Output:
      - spatial_features: [B, C_last, D', H', W'] for Grad-CAM
      - classical_embedding: [B, classical_embed_dim]
    """
    def __init__(self, in_channels=1, base_filters=32, classical_embed_dim=64):
        super().__init__()
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, base_filters), num_channels=base_filters),
            nn.GELU()
        )

        # Stage 1: [B, 32, 32, 32, 32]
        self.stage1 = ResidualBlock3D(base_filters, base_filters, stride=1)
        # Stage 2: [B, 64, 16, 16, 16]
        self.stage2 = ResidualBlock3D(base_filters, base_filters * 2, stride=2)
        # Stage 3: [B, 128, 8, 8, 8]
        self.stage3 = ResidualBlock3D(base_filters * 2, base_filters * 4, stride=2)
        
        # Self-Attention block at bottleneck
        self.attention = SpatialAttention3D(dim=base_filters * 4, num_heads=4)

        # Stage 4: [B, 128, 4, 4, 4]
        self.stage4 = ResidualBlock3D(base_filters * 4, base_filters * 4, stride=2)

        # Global pooling and embedding projection
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj = nn.Sequential(
            nn.Linear(base_filters * 4, classical_embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.attention(x)
        feat_map = self.stage4(x)  # For Grad-CAM hooking
        
        pooled = self.global_pool(feat_map).view(feat_map.size(0), -1)
        embed = self.proj(pooled)
        return feat_map, embed
