"""EIIE policy network pi_theta with Robust Asset Scoring (paper Sec. 2.2.4 and 4).

Per decision step t and asset m:

    h_{t,m}      = f(P_{t,m})                                  Step 1: asset embedding
    h~_{t,m}     = ReLU(W [h_{t,m}; h^pred_{t,m}] + b) + h_{t,m}   MAL (optional)
    z_{t,m}      = g([h~_{t,m}; w_{t,m}])                       Step 2: score generation
    z~_{t,m}     = z_{t,m} - Y if asset m is masked             ARM (training only)
    a_t          = softmax(z_{0,0}, z~_{t,1}, ..., z~_{t,M})    Step 3: allocation

z_{0,0} is the learnable cash bias. ARM itself happens in the data pipeline
(``ras.data.PortfolioDataset``), which marks masked assets in ``asset_mask``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ras.networks import build_extractor
from ras.predictor import normalize_patterns


def init_weights(module):
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def uses_holding(cfg):
    """Whether the holding w_{t,m} is concatenated to the embedding before g.

    Set by ``holding_in_score``; each released checkpoint records its setting.
    LSRE-CAAN never uses it.
    """
    return bool(cfg.get("holding_in_score")) and cfg["model"] != "lsre_caan"


class EIIEPolicy(nn.Module):
    def __init__(self, cfg, predictor=None):
        super().__init__()
        d = cfg["d"]
        self.f = build_extractor(cfg)
        self.holding_in_score = uses_holding(cfg)
        self.mask_constant = cfg["mask_constant"]
        self.fusion = nn.Sequential(nn.Conv1d(2 * d, d, kernel_size=1), nn.ReLU()) if cfg["mal"] else None
        self.g = nn.Conv1d(d + int(self.holding_in_score), 1, kernel_size=1)
        self.cash_bias = nn.Parameter(torch.zeros(self.cash_bias_shape))
        init_weights(self)
        self.predictor = None
        if cfg["mal"]:
            if predictor is None:
                raise ValueError("MAL requires a predictor")
            self.set_predictor(predictor)

    cash_bias_shape = (1, 1, 1)

    def set_predictor(self, predictor):
        """Attach the trained predictive network; it is frozen during policy learning."""
        predictor.requires_grad_(False)
        self.predictor = predictor

    def embed(self, patterns, asset_mask):
        """h~_{t,m}: [B, d, M]."""
        x = normalize_patterns(patterns)
        h = self.f(x, asset_mask)
        if h.dim() == 4:
            h = h[:, :, -1]
        if self.fusion is not None:
            h_pred = self.predictor.embed(patterns, asset_mask)
            h = self.fusion(torch.cat([h, h_pred], dim=1)) + h
        return h

    def score_assets(self, h, holding, asset_mask):
        """z~_{t,m} for m = 1..M: [B, M]."""
        if self.holding_in_score:
            h = torch.cat([h, holding[:, None, 1:]], dim=1)
        z = self.g(h)[:, 0]
        return torch.where(asset_mask, z, z - self.mask_constant)

    def allocate(self, z, asset_mask):
        scores = torch.cat([self.cash_bias.view(1, 1).expand(z.shape[0], 1), z], dim=1)
        return scores, F.softmax(scores, dim=-1)

    def forward(self, patterns, holding, asset_mask):
        """Return the scores (cash first) and the portfolio allocation a_t, both [B, M + 1]."""
        z = self.score_assets(self.embed(patterns, asset_mask), holding, asset_mask)
        scores, allocation = self.allocate(z, asset_mask)
        return {"scores": scores, "allocation": allocation}


class LSRECAANPolicy(EIIEPolicy):
    """LSRE-CAAN invests only in the top half of the assets ranked by score."""

    cash_bias_shape = (1,)

    def allocate(self, z, asset_mask):
        num_winners = int(z.shape[1] * 0.5)
        losers = torch.ones_like(z)
        losers.scatter_(1, torch.argsort(z, dim=-1)[:, -num_winners:], 0.0)
        scores = z - 1e9 * losers
        scores = torch.where(asset_mask, scores, scores - self.mask_constant)
        scores = torch.cat([self.cash_bias.view(1, 1).expand(z.shape[0], 1), scores], dim=1)
        return scores, F.softmax(scores, dim=-1)


def build_policy(cfg, predictor=None):
    cls = LSRECAANPolicy if cfg["model"] == "lsre_caan" else EIIEPolicy
    return cls(cfg, predictor)
