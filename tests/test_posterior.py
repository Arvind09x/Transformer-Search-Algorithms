import itertools
import math
from collections import defaultdict
from fractions import Fraction

import numpy as np
import pytest

from config import EnvConfig
from dp import dp_policy, hypergeom_rank, optimal_mask, val_state, val_uniform_table
from env import PAD, Observation, SearchEnv, rollout, sample_batch
from posterior import (mixture_one_step_policy, prior_table, rank_posterior, rank_prior, trap_prior, trap_weight,
                       uniform_prior)
from teachers import bracket, extreme_rank, value_feasible_bracket

OPEN = dict(n_min=1, cap_mode="linear")
SETTINGS = {  #(distribution, trap_strength, p) for the enumeration
    "uniform": ("uniform", 0.75, 0.1),
    "trap_s05": ("adversarial", 0.5, 0.1),
    "trap_s075": ("adversarial", 0.75, 0.1),
    "trap_s1": ("adversarial", 1.0, 0.1),
    "mixture_trap": ("mixture", 0.75, 0.3),
}


@pytest.fixture(scope="module")
def table():
    return val_uniform_table(16, 64)


def _cfg(n: int, v: int, setting: str) -> EnvConfig:
    distribution, s, p = SETTINGS[setting]
    return EnvConfig(v=v, n_min=n, n_max=n, feedback="val", distribution=distribution,
                     adversarial=None if distribution == "uniform" else "trap", trap_strength=s, p=p)


def _round_away(x: Fraction) -> int:
    return math.floor(x + Fraction(1, 2)) if x >= 0 else -math.floor(-x + Fraction(1, 2))


def _law(n: int, v: int, setting: str) -> dict[tuple[tuple[int, ...], int], Fraction]:
    distribution, s, p = SETTINGS[setting]
    p = Fraction(0) if distribution == "uniform" else Fraction(1) if distribution == "adversarial" else Fraction(str(p))
    s = Fraction(str(s))
    sets = list(itertools.combinations(range(v), n))
    base = Fraction(1, len(sets) * n)  #uniform: every (array, target index) pair
    by_rank: dict[tuple[int, int], list[tuple[int, ...]]] = defaultdict(list)
    for a in sets:
        for k in range(n):
            by_rank[(a[k], k)].append(a)
    law: dict[tuple[tuple[int, ...], int], Fraction] = defaultdict(Fraction)
    for (t, j0), arrays in by_rank.items():
        for a in arrays:
            law[(a, j0)] += (1 - p) * base
        if p == 0:
            continue
        r_lo, r_hi = max(0, n - v + t), min(n - 1, t)  #trap procedure, steps 1-5, from (T, j0) of a uniform draw
        k_ext = r_hi if 2 * t < v else r_lo
        edge = j0 + _round_away(s * (k_ext - j0))
        lo, hi = min(edge, k_ext), max(edge, k_ext)
        mass = p * base * len(arrays)
        for kp in range(lo, hi + 1):
            fill = by_rank[(t, kp)]  #step 5: uniform over arrays with T at rank kp
            for a in fill:
                law[(a, kp)] += mass / ((hi - lo + 1) * len(fill))
    assert sum(law.values()) == 1
    return {key: w for key, w in law.items() if w > 0}


def _brute_posteriors(n: int, v: int, setting: str) -> dict[tuple[int, frozenset], np.ndarray]:
    post: dict[tuple[int, frozenset], np.ndarray] = defaultdict(lambda: np.zeros(n))
    for (a, k), w in _law(n, v, setting).items():
        others = [i for i in range(n) if i != k]
        for size in range(n):
            for queried in itertools.combinations(others, size):  #every non-hit history this array is consistent with
                post[(a[k], frozenset((i, a[i]) for i in queried))][k] += float(w)
    return {key: x / x.sum() for key, x in post.items()}


def _observations(n: int, keys: list[tuple[int, frozenset]]) -> Observation:
    queries = np.full((len(keys), n), PAD, dtype=np.int64)
    feedback = np.full((len(keys), n), PAD, dtype=np.int64)
    for r, (_, history) in enumerate(keys):
        for s, (i, f) in enumerate(sorted(history)):
            queries[r, s], feedback[r, s] = i, f
    return Observation(n=np.full(len(keys), n), target=np.array([t for t, _ in keys]), queries=queries,
                       feedback=feedback, steps=(queries != PAD).sum(axis=1), done=np.zeros(len(keys), dtype=bool))


def _random_histories(cfg: EnvConfig, size: int, max_steps: int, seed: int) -> list[Observation]:
    rng = np.random.default_rng(seed)
    out = []
    for steps in range(max_steps + 1):
        env = SearchEnv(cfg, sample_batch(cfg, size, rng))
        for _ in range(steps):
            env.step(rng.integers(0, env.observe().n))  #repeats and out-of-bracket queries included
        out.append(env.observe())
    return out


@pytest.mark.parametrize("setting", list(SETTINGS))
@pytest.mark.parametrize("n, v", [(3, 8), (4, 10), (5, 12)])
def test_posterior_matches_enumeration(n: int, v: int, setting: str) -> None:
    brute = _brute_posteriors(n, v, setting)
    keys = list(brute)
    cfg = _cfg(n, v, setting)
    post = rank_posterior(_observations(n, keys), cfg, prior_table(cfg))
    want = np.array([brute[key] for key in keys])
    assert np.allclose(post, want, rtol=0, atol=1e-12)
    empty = [r for r, (_, h) in enumerate(keys) if not h]  #the empty history is the prior itself
    for r in empty:
        assert np.allclose(rank_prior(cfg, n, keys[r][0]), want[r], rtol=0, atol=1e-12)
    assert len(empty) > 0 and len(keys) > len(empty)


def test_p0_equals_hypergeometric() -> None:
    cfg = EnvConfig(v=64, n_max=16, feedback="val", **OPEN)
    prior = prior_table(cfg)
    checked = 0
    for obs in _random_histories(cfg, 200, 6, 0):
        post = rank_posterior(obs, cfg, prior)
        lo, m, size, t = val_state(obs, 64)
        for i in np.nonzero(~obs.done)[0]:
            inside = post[i, lo[i]:lo[i] + m[i]]
            assert np.allclose(inside, hypergeom_rank(int(m[i]), int(size[i]), int(t[i])), rtol=0, atol=1e-12)
            assert abs(inside.sum() - 1.0) < 1e-12
            checked += 1
    assert checked > 500


def test_one_step_equals_uniform_dp_at_p0(table) -> None:
    cfg = EnvConfig(v=64, n_max=16, feedback="val", **OPEN)
    policy, reference = mixture_one_step_policy(cfg, table), dp_policy(table, 64)
    env = SearchEnv(cfg, sample_batch(cfg, 1000, np.random.default_rng(1)))
    log: list = []

    def recording(obs: Observation) -> np.ndarray:
        log.append(obs)
        return policy(obs)
    rollout(env, recording)
    assert env.success.all()
    checked = 0
    for obs in log + _random_histories(cfg, 200, 6, 2):
        rows = np.nonzero(~obs.done)[0]
        action = policy(obs)
        assert np.array_equal(action[rows], reference(obs)[rows])
        assert optimal_mask(obs, table, 64)[rows, action[rows]].all()
        checked += len(rows)
    assert checked > 2000


@pytest.mark.parametrize("distribution, s, p", [("adversarial", 0.75, 0.1), ("adversarial", 1.0, 0.1),
                                                ("mixture", 0.75, 0.1), ("mixture", 0.75, 0.3)])
def test_actions_value_feasible_and_rollouts_succeed(table, distribution: str, s: float, p: float) -> None:
    cfg = EnvConfig(v=64, n_max=16, feedback="val", distribution=distribution, adversarial="trap",
                    trap_strength=s, p=p, **OPEN)
    policy = mixture_one_step_policy(cfg, table)
    env = SearchEnv(cfg, sample_batch(cfg, 2000, np.random.default_rng(3)))
    log: list = []

    def recording(obs: Observation) -> np.ndarray:
        action = policy(obs)
        log.append((obs, action))
        return action
    rollout(env, recording)
    assert env.success.all()
    states = log + [(obs, policy(obs)) for obs in _random_histories(cfg, 200, 6, 4)]
    checked = 0
    for obs, action in states:
        rows = np.nonzero(~obs.done)[0]
        vlo, vhi = value_feasible_bracket(obs, 64)
        assert np.all((vlo[rows] <= action[rows]) & (action[rows] <= vhi[rows]))
        checked += len(rows)
    assert checked > 2000


def test_priors() -> None:
    for v, n in ((64, 16), (64, 5), (256, 64)):
        for t in (0, 3, n - 1, v // 2 - 1, v // 2, v - n, v - 1):
            u = uniform_prior(n, t, v)
            assert abs(u.sum() - 1.0) < 1e-12 and np.all(u >= 0)
            for s in (0.0, 0.5, 0.75, 1.0):
                trap = trap_prior(n, t, v, s)
                assert abs(trap.sum() - 1.0) < 1e-12 and np.all(trap >= 0)
                assert np.all(trap[u == 0] == 0.0)  #windows lie inside the feasible ranks
            point = trap_prior(n, t, v, 1.0)
            k_ext = int(extreme_rank(np.array([n]), np.array([t]), v)[0])
            assert abs(point[k_ext] - 1.0) < 1e-12
    base = dict(v=64, n_max=16, feedback="val", **OPEN)
    assert trap_weight(EnvConfig(**base)) == 0.0
    assert trap_weight(EnvConfig(distribution="adversarial", adversarial="trap", **base)) == 1.0
    mix = EnvConfig(distribution="mixture", adversarial="trap", p=0.25, **base)
    assert trap_weight(mix) == 0.25
    table = prior_table(mix)
    for n, t in ((1, 0), (7, 30), (16, 63)):
        expected = 0.75 * uniform_prior(n, t, 64) + 0.25 * trap_prior(n, t, 64, mix.trap_strength)
        assert np.allclose(table[n, t, :n], expected, rtol=0, atol=1e-15) and np.all(table[n, t, n:] == 0.0)


def test_input_validation(table) -> None:
    base = dict(v=64, n_max=16, **OPEN)
    clustered = EnvConfig(feedback="val", distribution="mixture", adversarial="clustered", **base)
    with pytest.raises(ValueError):
        rank_prior(clustered, 8, 10)
    with pytest.raises(ValueError):
        mixture_one_step_policy(clustered, table)
    with pytest.raises(ValueError):
        mixture_one_step_policy(EnvConfig(**base), table)  #cmp
    with pytest.raises(ValueError):
        mixture_one_step_policy(EnvConfig(feedback="val", **base), val_uniform_table(8, 64))
    cmp_cfg = EnvConfig(**base)
    obs = SearchEnv(cmp_cfg, sample_batch(cmp_cfg, 5, np.random.default_rng(5))).observe()
    with pytest.raises(ValueError):
        rank_posterior(obs, EnvConfig(feedback="val", **base))
