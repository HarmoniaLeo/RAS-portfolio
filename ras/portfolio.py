"""Portfolio dynamics: immediate return R_t, reward log R_t and holding transition."""
import numpy as np
import torch


def cash_holding(batch_size, num_assets, device=None):
    """w_{t0} = (1, 0, ..., 0): all wealth in cash."""
    w = torch.zeros(batch_size, num_assets + 1, device=device)
    w[:, 0] = 1
    return w


def step(allocation, movement, holding, transaction_cost):
    """One decision step.

    Args:
        allocation: a_t, [B, M + 1] (cash first).
        movement: r_{t,1..M}, [B, M]; the cash has r_{t,0} = 1.
        holding: w_t, [B, M + 1].
    Returns:
        R_t, the portfolio's immediate return net of transaction costs [B], and
        the next holding w_{t+1} = (a_t * r_t) / (a_t^T r_t) [B, M + 1].
        The transaction cost c * ||a_t - w_t||_1 is charged proportionally to
        the gross return a_t^T r_t.
    """
    r = torch.cat([torch.ones_like(movement[:, :1]), movement], dim=1)
    gross = (allocation * r).sum(dim=-1)
    next_holding = allocation * r / gross[:, None]
    cost = transaction_cost * torch.abs(allocation - holding).sum(dim=-1)
    return gross * (1 - cost), next_holding


class PortfolioVectorMemory:
    """Holdings w_t indexed by trading day, shared by all training samples (as in EIIE).

    Training samples are drawn independently, so the holding of a sample is
    read from the memory and the resulting w_{t+1} is written back.
    """

    def __init__(self, num_days, num_assets):
        self.memory = np.zeros((num_days + 1, num_assets + 1), dtype=np.float32)
        self.memory[:, 0] = 1

    def read(self, t, restart_prob):
        w = torch.tensor(self.memory[t.numpy()])
        restart = torch.rand(len(t)) < restart_prob
        w[restart] = 0
        w[restart, 0] = 1
        return w

    def write(self, t, w):
        self.memory[t.numpy()] = w.detach().cpu().numpy()
