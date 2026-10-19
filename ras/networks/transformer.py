"""Transformer feature extractor of Transformer EIIE (paper Sec. 5.1).

Maps the price pattern P_{t,m} of every asset independently (shared weights) to
an embedding h_{t,m} in R^d:

    kernel-5 convolution over time -> N Transformer layers -> last token.

The same architecture is used for the predictive network f^pred of MAL.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, length, d):
        super().__init__()
        pe = torch.zeros(length, d)
        position = torch.arange(0.0, length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d, 2) * -(math.log(500.0) / length))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):  # [B*M, T, d]
        return x + self.pe[-x.shape[1]:]


class SelfAttention(nn.Module):
    def __init__(self, d, num_heads, length, causal=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.causal = causal
        if causal:
            self.register_buffer("causal_mask", torch.tril(torch.ones(length, length)).view(1, 1, length, length), persistent=False)

    def forward(self, x):  # [B*M, T, d]
        n, length, d = x.shape
        q, k, v = (proj(x).view(n, -1, self.num_heads, self.head_dim).transpose(1, 2)
                   for proj in (self.q_proj, self.k_proj, self.v_proj))
        scores = torch.matmul(q / math.sqrt(self.head_dim), k.transpose(-2, -1))
        if self.causal:
            scores = scores.masked_fill(self.causal_mask[:, :, :length, :length] == 0, -1e8)
        context = torch.matmul(F.softmax(scores, dim=-1), v)
        return self.out_proj(context.transpose(1, 2).contiguous().view(n, length, d))


class TransformerLayer(nn.Module):
    """Post-norm Transformer layer: self-attention and a position-wise FFN."""

    def __init__(self, d, num_heads, length, causal=False):
        super().__init__()
        self.self_attn = SelfAttention(d, num_heads, length, causal)
        self.norm1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        x = self.norm1(self.self_attn(x) + x)
        return self.norm2(self.ffn(x) + x)


class TransformerFeatureExtractor(nn.Module):
    """f (policy) or f^pred (MAL): P_{t,m} -> h_{t,m}.

    ``causal`` adds a causal attention mask. The results of the paper were
    produced with ``causal=False``; since only the last token is used, the
    stack then attends over the whole price pattern.
    """

    def __init__(self, in_channels=4, d=32, num_layers=2, num_heads=2, kernel_size=5, window=50, causal=False):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, d, kernel_size=(kernel_size, 1))
        self.positional_encoding = SinusoidalPositionalEncoding(window, d)
        self.layers = nn.ModuleList([TransformerLayer(d, num_heads, window, causal) for _ in range(num_layers)])

    def forward(self, patterns, asset_mask=None):  # [B, C, l, M] -> [B, d, M]
        x = self.input_conv(patterns)
        b, d, length, m = x.shape
        x = x.permute(0, 3, 2, 1).reshape(b * m, length, d)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return x[:, -1].view(b, m, d).permute(0, 2, 1).contiguous()
