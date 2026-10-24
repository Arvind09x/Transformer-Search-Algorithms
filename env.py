from dataclasses import dataclass
from typing import Callable

import numpy as np

from config import EnvConfig

LESS, EQUAL, GREATER = 0, 1, 2  #sign(A[i] - T) + 1; LESS means the target lies to the right
PAD = -1  #history and feedback padding; never a valid index, code, or value


def _readonly(*arrays: np.ndarray) -> None:
    for a in arrays:
        a.flags.writeable = False


@dataclass(frozen=True)
class EpisodeBatch:  #hidden state; never given to a policy
    n: np.ndarray  #(B,)
    values: np.ndarray  #(B, n_max) sorted, padded with v
    target_idx: np.ndarray  #(B,)
    target: np.ndarray  #(B,)
    adversarial: np.ndarray  #(B,) bool, row drawn from the adversarial component

    def __post_init__(self) -> None:
        _readonly(self.n, self.values, self.target_idx, self.target, self.adversarial)


def _pad_sorted(chosen: np.ndarray, n: np.ndarray, v: int, n_max: int) -> np.ndarray:
    cols = np.arange(chosen.shape[1])[None, :]
    return np.sort(np.where(cols < n[:, None], chosen, v), axis=1)[:, :n_max]


def _mask_to_values(mask: np.ndarray, v: int, n_max: int) -> np.ndarray:
    x = np.arange(v)[None, :]
    return np.sort(np.where(mask, x, v), axis=1)[:, :n_max]


def sample_n(cfg: EnvConfig, size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(cfg.n_min, cfg.n_max + 1, size=size)


def sample_uniform_values(n: np.ndarray, v: int, n_max: int, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permuted(np.tile(np.arange(v), (len(n), 1)), axis=1)  #first n entries of a uniform permutation
    return _pad_sorted(perm[:, :n_max], n, v, n_max)


def sample_clustered_centres(size: int, k: int, v: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, v, size=(size, k))


def clustered_log_weights(centres: np.ndarray, sigma: float, v: int) -> np.ndarray:
    x = np.arange(v)[None, None, :]
    z = -((x - centres[:, :, None]) ** 2) / (2.0 * sigma ** 2)  #(B, k, v)
    m = z.max(axis=1)
    return m + np.log(np.exp(z - m[:, None, :]).sum(axis=1))  #logsumexp over centres, (B, v)


def sample_clustered_values(n: np.ndarray, centres: np.ndarray, sigma: float, v: int, n_max: int,
                            rng: np.random.Generator) -> np.ndarray:
    keys = clustered_log_weights(centres, sigma, v) + rng.gumbel(size=(len(n), v))
    top = np.argsort(-keys, axis=1)  #Gumbel-top-k == sequential sampling without replacement, prop. to w
    return _pad_sorted(top[:, :n_max], n, v, n_max)


def round_half_away(x: np.ndarray) -> np.ndarray:
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(np.int64)


def trap_rank_window(n: np.ndarray, t: np.ndarray, j0: np.ndarray, v: int, s: float) -> tuple[np.ndarray, np.ndarray]:
    r_lo = np.maximum(0, n - v + t)
    r_hi = np.minimum(n - 1, t)
    k_ext = np.where(2 * t < v, r_hi, r_lo)  #T < V/2 in integer arithmetic
    edge = j0 + round_half_away(s * (k_ext - j0))
    return np.minimum(edge, k_ext), np.maximum(edge, k_ext)


def sample_trap_values(n: np.ndarray, t: np.ndarray, k_prime: np.ndarray, v: int, n_max: int,
                       rng: np.random.Generator) -> np.ndarray:
    keys = rng.random((len(n), v))  #one key per integer; the smallest keys on each side are a uniform subset
    x = np.arange(v)[None, :]
    below, above = x < t[:, None], x > t[:, None]
    chosen = x == t[:, None]
    for side, count in ((below, k_prime), (above, n - 1 - k_prime)):
        order = np.argsort(np.where(side, keys, np.inf), axis=1)
        take = np.zeros_like(side)
        np.put_along_axis(take, order, x < count[:, None], axis=1)
        chosen |= take
    return _mask_to_values(chosen, v, n_max)


def sample_batch(cfg: EnvConfig, size: int, rng: np.random.Generator) -> EpisodeBatch:
    if size < 1:
        raise ValueError(f"size={size} must be >= 1")
    rng_n, rng_flag, rng_arr = rng.spawn(3)
    v, n_max = cfg.v, cfg.n_max
    n = sample_n(cfg, size, rng_n)
    u = rng_flag.random(size)  #one uniform per row, so flagged sets are nested across p
    if cfg.distribution == "uniform":
        flag = np.zeros(size, dtype=bool)
    elif cfg.distribution == "adversarial":
        flag = np.ones(size, dtype=bool)
    else:
        flag = u < cfg.p
    values = sample_uniform_values(n, v, n_max, rng_arr)  #A0 and j0 for every row first
    target_idx = rng_arr.integers(0, n)
    rows = np.arange(size)
    if cfg.distribution != "uniform":  #component randomness for every row, kept only where flagged
        if cfg.adversarial == "clustered":
            centres = sample_clustered_centres(size, cfg.clustered_k, v, rng_arr)
            comp_values = sample_clustered_values(n, centres, cfg.clustered_sigma, v, n_max, rng_arr)
            comp_idx = rng_arr.integers(0, n)
        else:
            t = values[rows, target_idx]
            lo, hi = trap_rank_window(n, t, target_idx, v, cfg.trap_strength)
            comp_idx = rng_arr.integers(lo, hi + 1)
            comp_values = sample_trap_values(n, t, comp_idx, v, n_max, rng_arr)
        values = np.where(flag[:, None], comp_values, values)
        target_idx = np.where(flag, comp_idx, target_idx)
    return EpisodeBatch(n=n, values=values, target_idx=target_idx, target=values[rows, target_idx], adversarial=flag)


@dataclass(frozen=True)
class Observation:  #everything a policy may see
    n: np.ndarray  #(B,)
    target: np.ndarray | None  #None in cmp: T is absent, not masked
    queries: np.ndarray  #(B, k_max), PAD after steps
    feedback: np.ndarray  #(B, k_max), PAD after steps; LESS/EQUAL/GREATER in cmp, A[i] in val
    steps: np.ndarray  #(B,) queries made so far
    done: np.ndarray  #(B,)


class SearchEnv:
    def __init__(self, cfg: EnvConfig, batch: EpisodeBatch) -> None:
        if np.any(batch.n < 1) or np.any(batch.n > cfg.n_max) or batch.values.shape[1] < batch.n.max():
            raise ValueError("batch sizes inconsistent with config")
        self._cfg = cfg
        self._batch = batch
        size = len(batch.n)
        self.k_max = cfg.query_cap(cfg.n_max)  #history width, constant across batches
        cap_table = np.array([0] + [cfg.query_cap(m) for m in range(1, cfg.n_max + 1)])
        self._cap = cap_table[batch.n]
        self._queries = np.full((size, self.k_max), PAD, dtype=np.int64)
        self._feedback = np.full((size, self.k_max), PAD, dtype=np.int64)
        self._steps = np.zeros(size, dtype=np.int64)
        self._done = np.zeros(size, dtype=bool)
        self._success = np.zeros(size, dtype=bool)

    @property
    def cap(self) -> np.ndarray:
        return self._cap.copy()

    @property
    def queries_used(self) -> np.ndarray:
        return self._steps.copy()

    @property
    def done(self) -> np.ndarray:
        return self._done.copy()

    @property
    def success(self) -> np.ndarray:
        return self._success.copy()

    def observe(self) -> Observation:
        target = self._batch.target.copy() if self._cfg.feedback == "val" else None
        return Observation(n=self._batch.n.copy(), target=target, queries=self._queries.copy(),
                           feedback=self._feedback.copy(), steps=self._steps.copy(), done=self._done.copy())

    def step(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.asarray(idx)
        size = len(self._steps)
        if idx.shape != (size,) or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError(f"idx must be an integer array of shape ({size},)")
        rows = np.nonzero(~self._done)[0]  #finished rows ignore idx
        q = idx[rows].astype(np.int64)
        if np.any((q < 0) | (q >= self._batch.n[rows])):
            raise ValueError("query index outside [0, N) on an active row")
        a = self._batch.values[rows, q]
        t = self._batch.target[rows]
        fb = np.sign(a - t) + 1 if self._cfg.feedback == "cmp" else a
        feedback = np.full(size, PAD, dtype=np.int64)
        feedback[rows] = fb
        self._queries[rows, self._steps[rows]] = q
        self._feedback[rows, self._steps[rows]] = fb
        self._steps[rows] += 1
        reward = np.zeros(size, dtype=np.float64)
        reward[rows] = -self._cfg.query_cost
        hit = rows[a == t]
        self._success[hit] = True
        self._done[hit] = True
        capped = rows[(a != t) & (self._steps[rows] >= self._cap[rows])]
        self._done[capped] = True
        reward[capped] -= self._cfg.failure_cost_coef * self._cap[capped]
        return feedback, reward, self._done.copy()

    def cost(self) -> np.ndarray:
        if not self._done.all():
            raise ValueError("cost is defined only once every episode has ended")
        failed = ~self._success
        return self._cfg.query_cost * self._steps + self._cfg.failure_cost_coef * self._cap * failed


def rollout(env: SearchEnv, policy: Callable[[Observation], np.ndarray]) -> np.ndarray:
    rewards = np.zeros((len(env.done), env.k_max), dtype=np.float64)  #(B, k_max), reward of query t at column t
    for t in range(env.k_max):
        if env.done.all():
            break
        active = ~env.done
        _, r, _ = env.step(policy(env.observe()))
        rewards[active, t] = r[active]
    return rewards
