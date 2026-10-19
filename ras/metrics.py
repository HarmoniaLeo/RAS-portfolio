"""Evaluation metrics of the paper (Sec. 5.2).

All functions take the portfolio's immediate returns R_t (price ratios, so a
gain is R_t > 1) of consecutive decision steps.
"""
import numpy as np
from scipy.stats import spearmanr

TRADING_DAYS = 252


def periods_per_year(bar_days=5):
    return TRADING_DAYS // bar_days


def max_drawdown(R):
    """Maximum drawdown (negative number), identical to ``empyrical.max_drawdown``."""
    r = np.nan_to_num(np.asarray(R, dtype=np.float64) - 1)
    wealth = np.concatenate([[1.0], np.cumprod(1 + r)])
    peak = np.fmax.accumulate(wealth)
    return np.nanmin((wealth - peak) / peak)


def annual_return(R, periods=50):
    """AR: compound annual growth rate."""
    r = np.asarray(R, dtype=np.float64) - 1
    return np.prod(r + 1) ** (periods / len(r)) - 1


def annual_sharpe(R, periods=50, ddof=0):
    """ASR: annualised Sharpe ratio (zero risk-free rate).

    ``ddof`` of the standard deviation: model selection and ``metrics.json`` use
    ddof=0; the paper's tables were computed with the sample standard
    deviation (ddof=1), see ``ras.analysis``.
    """
    r = np.asarray(R, dtype=np.float64) - 1
    return np.sqrt(periods) * r.mean() / r.std(ddof=ddof)


def annual_calmar(R, periods=50):
    """ACR: annualised Calmar ratio, AR divided by the maximum drawdown."""
    return annual_return(R, periods) / -max_drawdown(R)


def train_reward(R):
    """Train Reward: average reward log R_t over all decision steps of the train set."""
    return float(np.mean(np.log(np.asarray(R, dtype=np.float64))))


def performance(R, periods=50):
    return {"AR": annual_return(R, periods), "ASR": annual_sharpe(R, periods), "ACR": annual_calmar(R, periods),
            "MDD": -max_drawdown(R), "MeanLogReturn": train_reward(R)}


def rank_ic_series(scores, movements):
    """Rank IC_t: Spearman correlation of the scores z_t and price movements r_t at every step."""
    return np.array([spearmanr(movements[t], scores[t])[0] for t in range(len(scores))])


def rank_ic(scores, movements):
    """Mean Rank IC over all steps."""
    return float(np.nanmean(rank_ic_series(scores, movements)))


def rank_ic_ir(scores, movements):
    """Rank IC information ratio mean(Rank IC_t) / std(Rank IC_t).

    This is the quantity reported in the "Rank IC" columns of the paper's
    ablation tables and of the standalone evaluation of MAL.
    """
    ic = rank_ic_series(scores, movements)
    return float(np.nanmean(ic) / np.nanstd(ic))


def pearson(scores, movements):
    return float(np.corrcoef(np.ravel(movements), np.ravel(scores))[1, 0])


def mean_bias(scores, movements):
    return float(np.mean(np.ravel(scores) - np.ravel(movements)))


def cohens_d(scores, movements, pooled=False):
    """Standardised mean bias.

    ``pooled=False`` divides the mean bias by the standard deviation of the price
    movement, which is how the values in the paper were computed; ``pooled=True``
    uses the pooled standard deviation sqrt((sigma_z^2 + sigma_r^2) / 2).
    """
    z, r = np.ravel(scores), np.ravel(movements)
    scale = np.sqrt((z.std() ** 2 + r.std() ** 2) / 2) if pooled else r.std()
    return float(np.mean(z - r) / (scale + 1e-12))
