import math
from dataclasses import dataclass

import numpy as np

SNAP_TOL = 1e-9  #alpha * B within this of an integer is that integer: 0.07 * 100 = 7.000000000000001 is 7


@dataclass(frozen=True)
class TailStats:
    nu: float  #upper alpha-quantile of the costs: the ceil(alpha B)-th largest
    cvar: float
    k: float  #alpha * B, the tail mass in episodes
    strictly_worse: int  #count of C > nu: the episodes with nonzero surrogate weight
    at_quantile: int  #count of C == nu
    zero_weight: bool  #no episode strictly worse than nu: the surrogate gradient is exactly zero
    tail: np.ndarray  #(B,) bool, C >= nu: the tail including the boundary atom
    nonzero: np.ndarray  #(B,) bool, C > nu: the nonzero-weight set


def tail_size(alpha: float, size: int) -> float:
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha={alpha} must lie in (0, 1]")
    if size < 1:
        raise ValueError("need at least one episode")
    k = alpha * size
    return float(round(k)) if abs(k - round(k)) <= SNAP_TOL * max(1.0, k) else k


def _costs(costs: np.ndarray) -> np.ndarray:
    costs = np.asarray(costs, dtype=np.float64)
    if costs.ndim != 1 or costs.size == 0:
        raise ValueError(f"costs must be a non-empty 1-D array, got shape {costs.shape}")
    if not np.all(np.isfinite(costs)):
        raise ValueError("costs must be finite")
    return costs


def _sorted_desc(costs: np.ndarray) -> np.ndarray:
    return -np.sort(-costs)


def quantile(costs: np.ndarray, alpha: float) -> float:
    costs = _costs(costs)  #costs C, higher is worse (Appendix B); returns R = -C have nu_R = -nu
    k = tail_size(alpha, costs.size)
    return float(_sorted_desc(costs)[math.ceil(k) - 1])


def cvar(costs: np.ndarray, alpha: float) -> float:
    costs = _costs(costs)
    k = tail_size(alpha, costs.size)
    desc = _sorted_desc(costs)
    whole = math.floor(k)
    boundary = (k - whole) * desc[whole] if whole < costs.size else 0.0  #fractional mass on the boundary atom
    return float((desc[:whole].sum() + boundary) / k)


def surrogate_weights(costs: np.ndarray, alpha: float) -> np.ndarray:
    costs = _costs(costs)
    k = tail_size(alpha, costs.size)
    nu = quantile(costs, alpha)
    return np.where(costs > nu, (nu - costs) / k, 0.0)  #(R - nu_R) 1[R <= nu_R] / (alpha B), R = -C; ties at nu get 0


def tail_stats(costs: np.ndarray, alpha: float) -> TailStats:
    costs = _costs(costs)
    nu = quantile(costs, alpha)
    tail, nonzero = costs >= nu, costs > nu
    worse = int(nonzero.sum())
    return TailStats(nu=nu, cvar=cvar(costs, alpha), k=tail_size(alpha, costs.size), strictly_worse=worse,
                     at_quantile=int((costs == nu).sum()), zero_weight=worse == 0, tail=tail, nonzero=nonzero)
