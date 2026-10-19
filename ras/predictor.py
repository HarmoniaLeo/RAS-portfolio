"""Predictive network of Model-Augmented Learning (paper Sec. 4.3).

    h^pred_{t,m} = f^pred(P_{t,m})          predictive embedding (Transformer, d_pred = d)
    z^_{t,m}     = g^pred(h^pred_{t,m})      predicted score, an estimate of r_{t,m}

Trained alone by minimising the MSE loss L_pred against the realised price
movement, then frozen and used inside the policy (see ``ras.policy``).
"""
import torch
import torch.nn as nn

from ras.networks import build_extractor


def normalize_patterns(patterns):
    """Divide every observation of P_{t,m} by the last closing price p^C_{t-1,m}."""
    return patterns / (patterns[:, 0:1, -1:, :] + 1e-8)


class PricePredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.f_pred = build_extractor(cfg, "predictor")
        self.g_pred = nn.Conv1d(cfg["d"], 1, kernel_size=1)  # 1x1 convolution = position-wise linear map
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def embed(self, patterns, asset_mask=None):
        """h^pred_{t,m}: [B, d_pred, M]."""
        return self.f_pred(normalize_patterns(patterns), asset_mask)

    def forward(self, patterns, asset_mask=None):
        """Predicted score z^_{t,m} (a price ratio): [B, M]."""
        return self.g_pred(self.embed(patterns, asset_mask))[:, 0] + 1


def prediction_loss(z_hat, movement, asset_mask, clip=0.1):
    """L_pred: squared error between z^_{t,m} and r_{t,m}, clipped at ``clip`` per sample for stability."""
    loss = (z_hat - movement).pow(2)
    loss = torch.where(asset_mask, loss, torch.zeros_like(loss))
    return torch.clamp(loss, min=0.0, max=clip).mean()
