import functools
import math

import numpy as np

from config import EnvConfig
from dp import TIE_TOL, ValTable, hypergeom_rank, val_continuation, val_outcome_given_rank
from env import Observation, trap_rank_window
from teachers import Policy, bracket, value_bounds

G_CACHE = 4096  #(m, L, t) continuation matrices kept, at most m * m * 8 bytes each


@functools.lru_cache(maxsize=None)
def _log_binom_table(n_max: int) -> np.ndarray:
    table = np.full((n_max + 1, n_max + 1), -np.inf)
    for n in range(n_max + 1):
        for k in range(n + 1):
            table[n, k] = math.log(math.comb(n, k))  #exact integer, then one rounding
    return table


def _log_binom(n: np.ndarray, k: np.ndarray, v: int) -> np.ndarray:
    table = _log_binom_table(v)
    n, k = np.asarray(n), np.asarray(k)
    valid = (n >= 0) & (k >= 0) & (k <= n)
    return np.where(valid, table[np.clip(n, 0, v), np.clip(k, 0, v)], -np.inf)


def uniform_prior(n: int, t: int, v: int) -> np.ndarray:
    k = np.arange(n)
    return np.exp(_log_binom(t, k, v) + _log_binom(v - t - 1, n - 1 - k, v) - _log_binom(v - 1, n - 1, v))  #P(k | N, T)


def trap_prior(n: int, t: int, v: int, s: float) -> np.ndarray:
    j0 = np.arange(n)
    weight = uniform_prior(n, t, v)  #P_uniform(j0 | N, T): (N, T) has the uniform law under trap
    lo, hi = trap_rank_window(np.full(n, n), np.full(n, t), j0, v, s)
    k = np.arange(n)[None, :]
    inside = (lo[:, None] <= k) & (k <= hi[:, None])
    return (weight / (hi - lo + 1)) @ inside  #sum over j0 of P(j0) * 1[k in window(j0)] / |window(j0)|; exact zeros elsewhere


def trap_weight(cfg: EnvConfig) -> float:
    if cfg.distribution == "uniform":
        return 0.0
    if cfg.adversarial != "trap":
        raise ValueError(f"no rank posterior for adversarial={cfg.adversarial!r}; only trap is uniform given the rank")
    return 1.0 if cfg.distribution == "adversarial" else cfg.p  #P(trap | N, T) = p: (N, T) has the same law in both


def rank_prior(cfg: EnvConfig, n: int, t: int) -> np.ndarray:
    p = trap_weight(cfg)
    prior = (1.0 - p) * uniform_prior(n, t, cfg.v)
    if p > 0.0:
        prior = prior + p * trap_prior(n, t, cfg.v, cfg.trap_strength)
    return prior


def prior_table(cfg: EnvConfig) -> np.ndarray:
    table = np.zeros((cfg.n_max + 1, cfg.v, cfg.n_max))  #[N, T, k]
    for n in range(cfg.n_min, cfg.n_max + 1):
        for t in range(cfg.v):
            table[n, t, :n] = rank_prior(cfg, n, t)
    return table


def rank_posterior(obs: Observation, cfg: EnvConfig, prior: np.ndarray | None = None) -> np.ndarray:
    v = cfg.v
    lo, hi = bracket(obs)
    a, b = value_bounds(obs, v)  #raises under cmp
    if prior is None:
        prior = prior_table(cfg)
    n, t = obs.n[:, None], obs.target[:, None]
    lo, hi, a, b = lo[:, None], hi[:, None], a[:, None], b[:, None]
    k = np.arange(prior.shape[2])[None, :]
    terms = (_log_binom(t - 1 - a, k - lo, v), _log_binom(t, k, v),  #below T: last gap given rank k, over all k-subsets
             _log_binom(b - t - 1, hi - k, v), _log_binom(v - t - 1, n - 1 - k, v))  #above T, symmetric
    with np.errstate(divide="ignore"):
        log_prior = np.log(prior[obs.n, obs.target])
    valid = (lo <= k) & (k <= hi) & np.isfinite(log_prior) & np.all([np.isfinite(x) for x in terms], axis=0)
    below, total_below, above, total_above = (np.where(valid, x, 0.0) for x in terms)
    log_post = np.where(valid, np.where(valid, log_prior, 0.0) + below - total_below + above - total_above, -np.inf)
    log_post -= log_post.max(axis=1, keepdims=True)
    post = np.exp(log_post)
    return post / post.sum(axis=1, keepdims=True)  #(B, n_max) over absolute indices; zero outside [lo, hi]


def mixture_one_step_policy(cfg: EnvConfig, table: ValTable) -> Policy:
    if cfg.feedback != "val":
        raise ValueError("the mixture one-step DP needs feedback='val'")
    if table.n_max < cfg.n_max or table.l_max < cfg.v:
        raise ValueError(f"table (n_max={table.n_max}, l_max={table.l_max}) does not cover (n_max={cfg.n_max}, v={cfg.v})")
    prior = prior_table(cfg)  #also rejects clustered

    @functools.lru_cache(maxsize=G_CACHE)
    def continuation(m: int, size: int, t: int) -> np.ndarray:
        g = np.empty((m, m))  #g[j - 1, k - 1] = E[V_uniform(successor) | query position j, rank k]
        feasible = hypergeom_rank(m, size, t) > 0  #ranks T can take given (m, L, t); the posterior is zero elsewhere
        for j in range(1, m + 1):
            cond = val_outcome_given_rank(m, size, t, j)
            cont = val_continuation(table, m, size, t, j)
            joint = np.where(feasible[:, None], cond, 0.0)
            g[j - 1] = np.where(joint > 0, joint * cont[None, :], 0.0).sum(axis=1)
        return g

    def policy(obs: Observation) -> np.ndarray:
        post = rank_posterior(obs, cfg, prior)
        lo, hi = bracket(obs)
        a, b = value_bounds(obs, cfg.v)
        action = np.zeros(len(obs.n), dtype=np.int64)
        for i in np.nonzero(~obs.done)[0]:
            l, h = int(lo[i]), int(hi[i])
            q = 1.0 + continuation(h - l + 1, int(b[i] - a[i] - 1), int(obs.target[i] - a[i])) @ post[i, l:h + 1]
            action[i] = l + int(np.argmax(q <= q.min() + TIE_TOL))  #lowest position within the tie tolerance
        return action
    return policy
