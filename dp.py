import functools
import itertools
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from env import EQUAL, GREATER, LESS, Observation
from teachers import Policy, bracket, value_bounds

TIE_TOL = 1e-9  #absolute tolerance on Q for float argmin sets; Q lies in [1, 64]

History = frozenset[tuple[int, int]]  #set of (index, feedback) pairs; order does not change the state
Enumeration = dict[History, tuple[Fraction, tuple[Fraction, ...]]]  #history -> (V, Q for every index)


@dataclass(frozen=True)
class CmpTable:
    total: np.ndarray  #(n_max + 1,) int64, S(m) = m * V(m), exact


@dataclass(frozen=True)
class ValTable:
    value: np.ndarray  #(n_max + 1, l_max + 1, l_max + 1) V[m, L, t], NaN off the support (m = 0, L < m, t outside [1, L])
    best: np.ndarray  #same shape, int16: lowest optimal bracket offset j - 1, -1 off the support
    n_max: int
    l_max: int


@functools.lru_cache(maxsize=None)
def _binom_table(n_max: int) -> np.ndarray:
    table = np.zeros((n_max + 1, n_max + 1), dtype=np.float64)
    for n in range(n_max + 1):
        for k in range(n + 1):
            table[n, k] = float(math.comb(n, k))  #exact integer, rounded once
    return table


def _binoms(n: int) -> np.ndarray:
    return _binom_table(1 << n.bit_length())  #power-of-two sizes keep the cache to a few tables


def _binom(table: np.ndarray, n: np.ndarray, k: np.ndarray) -> np.ndarray:
    n, k = np.asarray(n), np.asarray(k)
    valid = (n >= 0) & (k >= 0) & (k <= n)
    top = table.shape[0] - 1
    return np.where(valid, table[np.clip(n, 0, top), np.clip(k, 0, top)], 0.0)


def cmp_table(n_max: int) -> CmpTable:
    if n_max < 1:
        raise ValueError(f"n_max={n_max} must be >= 1")
    total = [0, 1]
    for m in range(2, n_max + 1):
        total.append(m + min(total[j - 1] + total[m - j] for j in range(1, m + 1)))
    return CmpTable(total=np.array(total, dtype=np.int64))


def cmp_value(table: CmpTable, m: np.ndarray) -> np.ndarray:
    m = np.asarray(m)
    return table.total[m] / np.maximum(m, 1)


def cmp_q_totals(table: CmpTable, m: int) -> np.ndarray:
    j = np.arange(1, m + 1)
    return m + table.total[j - 1] + table.total[m - j]  #m * Q(j), exact; Q(j) = 1 + ((j-1) V(j-1) + (m-j) V(m-j)) / m


def cmp_q(table: CmpTable, m: int) -> np.ndarray:
    return cmp_q_totals(table, m) / m


def val_uniform_table(n_max: int, l_max: int) -> ValTable:
    if not 1 <= n_max <= l_max:
        raise ValueError(f"need 1 <= n_max={n_max} <= l_max={l_max}")
    binom = _binoms(l_max)
    shape = (n_max + 1, l_max + 1, l_max + 1)
    value = np.full(shape, np.nan)
    best = np.full(shape, -1, dtype=np.int16)
    for size in range(1, l_max + 1):
        value[1, size, 1:size + 1] = 1.0  #one unknown: it is T
        best[1, size, 1:size + 1] = 0
    for m in range(2, n_max + 1):
        j = np.arange(1, m + 1)[:, None, None]
        for size in range(m, l_max + 1):
            t = np.arange(1, size + 1)[None, :, None]
            u = np.arange(1, size + 1)[None, None, :]  #observed value a + u
            denom = binom[size - 1, m - 1]
            p_less = _binom(binom, u - 1, j - 1) * _binom(binom, size - u - 1, m - j - 1) / denom  #A[j] = a + u < T
            p_greater = _binom(binom, u - 2, j - 2) * _binom(binom, size - u, m - j) / denom  #A[j] = a + u > T
            less = (u < t) & (p_less > 0)
            greater = (u > t) & (p_greater > 0)
            v_less = value[m - j, size - u, np.clip(t - u, 0, None)]  #successor (m - j, L - u, t - u)
            v_greater = value[j - 1, u - 1, t]  #successor (j - 1, u - 1, t)
            cont = (np.where(less, p_less * v_less, 0.0).sum(axis=2)
                    + np.where(greater, p_greater * v_greater, 0.0).sum(axis=2))  #(m, L) over (j, t)
            low = cont.min(axis=0)
            value[m, size, 1:size + 1] = 1.0 + low
            best[m, size, 1:size + 1] = np.argmax(cont <= low + TIE_TOL, axis=0)
    return ValTable(value=value, best=best, n_max=n_max, l_max=l_max)


def save_val_table(table: ValTable, path: str | Path) -> None:
    np.savez_compressed(path, value=table.value, best=table.best, n_max=table.n_max, l_max=table.l_max)


def load_val_table(path: str | Path) -> ValTable:
    with np.load(path) as f:
        table = ValTable(value=f["value"], best=f["best"], n_max=int(f["n_max"]), l_max=int(f["l_max"]))
    shape = (table.n_max + 1, table.l_max + 1, table.l_max + 1)
    if table.value.shape != shape or table.best.shape != shape:
        raise ValueError(f"{path}: arrays {table.value.shape}, {table.best.shape} do not match n_max={table.n_max}, "
                         f"l_max={table.l_max}")
    return table


def hypergeom_rank(m: int, size: int, t: int) -> np.ndarray:
    binom = _binoms(size)
    k = np.arange(1, m + 1)
    return _binom(binom, t - 1, k - 1) * _binom(binom, size - t, m - k) / binom[size - 1, m - 1]  #P(rank k) under uniform


def val_outcome_given_rank(m: int, size: int, t: int, j: int) -> np.ndarray:
    binom = _binoms(size)
    k = np.arange(1, m + 1)[:, None]
    u = np.arange(size + 1)[None, :]
    w = u - t
    below = _binom(binom, u - 1, j - 1) * _binom(binom, t - 1 - u, k - 1 - j) / _binom(binom, t - 1, k - 1).clip(1.0)
    above = _binom(binom, w - 1, j - k - 1) * _binom(binom, size - t - w, m - j) / _binom(binom, size - t, m - k).clip(1.0)
    out = np.where((k > j) & (u >= 1) & (u < t), below, 0.0) + np.where((k < j) & (u > t), above, 0.0)
    return out  #(m, L + 1): row k - 1, column u = P(A[j] = a + u | rank k); row j - 1 is the hit and is zero


def val_continuation(table: ValTable, m: int, size: int, t: int, j: int) -> np.ndarray:
    u = np.arange(size + 1)
    cont = np.full(size + 1, np.nan)
    lo_side, hi_side = (u >= 1) & (u < t), u > t
    cont[lo_side] = table.value[m - j, size - u[lo_side], t - u[lo_side]]
    cont[hi_side] = table.value[j - 1, u[hi_side] - 1, t]
    return cont  #(L + 1,): V of the successor after observing a + u; NaN where no successor exists


def val_q_given_rank(table: ValTable, m: int, size: int, t: int, rank_probs: np.ndarray) -> np.ndarray:
    if m > table.n_max or size > table.l_max:
        raise ValueError(f"state (m={m}, L={size}) outside table (n_max={table.n_max}, l_max={table.l_max})")
    q = np.empty(m)
    for j in range(1, m + 1):
        joint = rank_probs[:, None] * val_outcome_given_rank(m, size, t, j)
        cont = val_continuation(table, m, size, t, j)
        q[j - 1] = 1.0 + np.where(joint > 0, joint * cont[None, :], 0.0).sum()
    return q  #Q of bracket position j at index j - 1; rank_probs over ranks 1..m need not be the uniform posterior


def val_q(table: ValTable, m: int, size: int, t: int) -> np.ndarray:
    return val_q_given_rank(table, m, size, t, hypergeom_rank(m, size, t))


def cmp_state(obs: Observation) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = bracket(obs)
    return lo, hi - lo + 1


def val_state(obs: Observation, v: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = bracket(obs)
    a, b = value_bounds(obs, v)
    return lo, hi - lo + 1, b - a - 1, obs.target - a


def _open(obs: Observation, n_index: int) -> np.ndarray:
    if obs.n.max() > n_index:
        raise ValueError(f"N={obs.n.max()} exceeds n_index={n_index}")
    return np.nonzero(~obs.done)[0]


def q_values(obs: Observation, table: CmpTable | ValTable, v: int, n_index: int = 64) -> np.ndarray:
    rows = _open(obs, n_index)
    q = np.full((len(obs.n), n_index), np.nan)  #done rows stay NaN
    idx = np.arange(n_index)
    if isinstance(table, CmpTable):
        lo, m = cmp_state(obs)
        for i in rows:
            inside = cmp_q(table, int(m[i]))
            row = np.full(n_index, 1.0 + cmp_value(table, m[i]))  #repeat or out-of-bracket: no information
            row[lo[i]:lo[i] + m[i]] = inside
            q[i] = np.where(idx < obs.n[i], row, np.inf)
        return q
    lo, m, size, t = val_state(obs, v)
    for i in rows:
        inside = val_q(table, int(m[i]), int(size[i]), int(t[i]))
        row = np.full(n_index, 1.0 + table.value[m[i], size[i], t[i]])
        row[lo[i]:lo[i] + m[i]] = inside
        q[i] = np.where(idx < obs.n[i], row, np.inf)
    return q


def optimal_mask(obs: Observation, table: CmpTable | ValTable, v: int, n_index: int = 64,
                 tol: float = TIE_TOL) -> np.ndarray:
    rows = _open(obs, n_index)
    mask = np.zeros((len(obs.n), n_index), dtype=bool)
    if isinstance(table, CmpTable):
        lo, m = cmp_state(obs)
        for i in rows:
            totals = cmp_q_totals(table, int(m[i]))
            mask[i, lo[i]:lo[i] + m[i]] = totals == totals.min()  #exact integer ties
        return mask
    q = q_values(obs, table, v, n_index)
    mask[rows] = q[rows] <= q[rows].min(axis=1, keepdims=True) + tol
    return mask


def dp_policy(table: CmpTable | ValTable, v: int, n_index: int = 64) -> Policy:
    if isinstance(table, CmpTable):
        def cmp_policy(obs: Observation) -> np.ndarray:
            return np.argmax(optimal_mask(obs, table, v, n_index), axis=1)  #lowest optimal index; 0 on done rows
        return cmp_policy

    def val_policy(obs: Observation) -> np.ndarray:
        rows = _open(obs, n_index)
        lo, m, size, t = val_state(obs, v)
        if np.any(m[rows] > table.n_max) or np.any(size[rows] > table.l_max):
            raise ValueError("state outside the table")
        action = np.zeros(len(obs.n), dtype=np.int64)
        action[rows] = lo[rows] + table.best[m[rows], size[rows], t[rows]]
        return action
    return val_policy


def enumerate_cmp(n: int) -> Enumeration:
    memo: dict[frozenset[int], tuple[Fraction, tuple[Fraction, ...]]] = {}

    def code(i: int, k: int) -> int:
        return LESS if i < k else EQUAL if i == k else GREATER

    def solve(state: frozenset[int]) -> tuple[Fraction, tuple[Fraction, ...]]:
        if state in memo:
            return memo[state]
        q: list[Fraction | None] = [None] * n
        for i in range(n):
            groups: dict[int, list[int]] = {}
            for k in state:
                groups.setdefault(code(i, k), []).append(k)
            if set(groups) == {LESS} or set(groups) == {GREATER}:
                continue  #outcome known in advance: no information
            q[i] = 1 + sum((Fraction(len(g), len(state)) * solve(frozenset(g))[0]
                            for c, g in groups.items() if c != EQUAL), Fraction(0))
        low = min(x for x in q if x is not None)
        memo[state] = (low, tuple(x if x is not None else 1 + low for x in q))
        return memo[state]

    out: Enumeration = {}

    def walk(history: History, state: frozenset[int]) -> None:
        if history in out:
            return
        out[history] = solve(state)
        for i in range(n):
            groups: dict[int, list[int]] = {}
            for k in state:
                groups.setdefault(code(i, k), []).append(k)
            if len(groups) > 1 or EQUAL in groups:
                for c, g in groups.items():
                    if c != EQUAL:
                        walk(history | {(i, c)}, frozenset(g))

    walk(frozenset(), frozenset(range(n)))
    return out


def enumerate_val_uniform(n: int, v: int, target: int) -> Enumeration:
    arrays = frozenset(s for s in itertools.combinations(range(v), n) if target in s)  #equally likely given T
    if not arrays:
        raise ValueError(f"no array of {n} distinct values in [0, {v}) contains {target}")
    memo: dict[frozenset[tuple[int, ...]], tuple[Fraction, tuple[Fraction, ...]]] = {}

    def split(state: frozenset[tuple[int, ...]], i: int) -> dict[int, list[tuple[int, ...]]]:
        groups: dict[int, list[tuple[int, ...]]] = {}
        for s in state:
            groups.setdefault(s[i], []).append(s)
        return groups

    def informative(groups: dict[int, list[tuple[int, ...]]]) -> bool:
        return len(groups) > 1 or target in groups

    def solve(state: frozenset[tuple[int, ...]]) -> tuple[Fraction, tuple[Fraction, ...]]:
        if state in memo:
            return memo[state]
        q: list[Fraction | None] = [None] * n
        for i in range(n):
            groups = split(state, i)
            if informative(groups):
                q[i] = 1 + sum((Fraction(len(g), len(state)) * solve(frozenset(g))[0]
                                for f, g in groups.items() if f != target), Fraction(0))
        low = min(x for x in q if x is not None)
        memo[state] = (low, tuple(x if x is not None else 1 + low for x in q))
        return memo[state]

    out: Enumeration = {}

    def walk(history: History, state: frozenset[tuple[int, ...]]) -> None:
        if history in out:
            return
        out[history] = solve(state)
        for i in range(n):
            groups = split(state, i)
            if informative(groups):
                for f, g in groups.items():
                    if f != target:
                        walk(history | {(i, f)}, frozenset(g))

    walk(frozenset(), arrays)
    return out
