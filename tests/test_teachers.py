import dataclasses
import itertools
import math
from fractions import Fraction
from typing import Callable

import numpy as np
import pytest

from config import EnvConfig
from env import EQUAL, GREATER, LESS, PAD, EpisodeBatch, Observation, SearchEnv, rollout, sample_batch, trap_rank_window
from teachers import (ALWAYS_FEASIBLE, POLICY_NAMES, VALUE_BASED, adaptive_switch, binary_search, bracket,
                      comparison_codes, extreme_rank, guarded_interpolation, interpolate_k_then_vf_midpoint,
                      interpolate_once, interpolation_search, make_policy, probe_extreme, value_bounds,
                      value_feasible_bracket, value_feasible_midpoint, value_feasible_widths)

V = 256
OPEN = dict(n_min=1, cap_mode="linear")  #K(N) = N: a policy that stays in the bracket and never repeats cannot be capped
SETTINGS = {
    "uniform": EnvConfig(feedback="val", **OPEN),
    "clustered": EnvConfig(feedback="val", distribution="adversarial", adversarial="clustered", **OPEN),
    "trap_s1": EnvConfig(feedback="val", distribution="adversarial", adversarial="trap", trap_strength=1.0, **OPEN),
    "trap_s075": EnvConfig(feedback="val", distribution="adversarial", adversarial="trap", **OPEN),
    "mixture_trap": EnvConfig(feedback="val", distribution="mixture", adversarial="trap", p=0.5, **OPEN),
}
SMALL = {
    "uniform": EnvConfig(v=12, n_max=6, feedback="val", **OPEN),
    "mixture_trap": EnvConfig(v=12, n_max=6, feedback="val", distribution="mixture", adversarial="trap", p=0.5, **OPEN),
}


def _batch(rows: list[list[int]], target_idx: list[int], v: int = V) -> EpisodeBatch:
    n = np.array([len(r) for r in rows])
    values = np.full((len(rows), int(n.max())), v, dtype=np.int64)
    for i, r in enumerate(rows):
        values[i, :len(r)] = r
    t = np.asarray(target_idx, dtype=np.int64)
    return EpisodeBatch(n=n, values=values, target_idx=t, target=values[np.arange(len(rows)), t],
                        adversarial=np.zeros(len(rows), dtype=bool))


def _observe_after(cfg: EnvConfig, rows: list[list[int]], target_idx: list[int], queries: list[int]) -> Observation:
    env = SearchEnv(cfg, _batch(rows, target_idx, cfg.v))
    for q in queries:
        env.step(np.full(len(rows), q))
    return env.observe()


def _histories(cfg: EnvConfig, size: int, max_steps: int, seed: int) -> list[tuple[EpisodeBatch, Observation]]:
    rng = np.random.default_rng(seed)
    out = []
    for steps in range(max_steps + 1):  #uniform random queries: repeats and out-of-bracket queries included
        batch = sample_batch(cfg, size, rng)
        env = SearchEnv(cfg, batch)
        for _ in range(steps):
            env.step(rng.integers(0, batch.n))
        out.append((batch, env.observe()))
    return out


def _brute(batch: EpisodeBatch, obs: Observation, i: int, v: int) -> tuple[int, int, int, int]:
    n, j, t = int(batch.n[i]), int(batch.target_idx[i]), int(batch.target[i])
    row = batch.values[i].tolist()
    qs = obs.queries[i, :obs.steps[i]].tolist()
    assert all(row[q] != t for q in qs)  #open rows only: no EQUAL in the history
    consistent = [k for k in range(n) if all(k > q if row[q] < t else k < q for q in qs)]
    lo, hi = consistent[0], consistent[-1]
    assert consistent == list(range(lo, hi + 1)) and lo <= j <= hi
    a = row[lo - 1] if lo > 0 else -1
    b = row[hi + 1] if hi < n - 1 else v
    return lo, hi, a, b


def _open_rows(obs: Observation) -> np.ndarray:
    return np.nonzero(~obs.done)[0]


def _extend(cfg: EnvConfig, batch: EpisodeBatch, obs: Observation, action: np.ndarray) -> Observation:
    env = SearchEnv(cfg, batch)
    for s in range(int(obs.steps.max())):
        env.step(np.where(s < obs.steps, obs.queries[:, s], 0))  #rows with fewer steps are done; their index is ignored
    assert np.array_equal(env.observe().queries, obs.queries)
    env.step(action)
    return env.observe()


def _width(obs: Observation) -> np.ndarray:
    vlo, vhi = value_feasible_bracket(obs, V)
    return vhi - vlo + 1


def _hypergeom_mean(m: int, size: int, t: int) -> Fraction:
    total = math.comb(size - 1, m - 1)  #other m - 1 values drawn from the L - 1 integers other than T
    return sum((Fraction(k * math.comb(t - 1, k) * math.comb(size - t, m - 1 - k), total) for k in range(m)),
               Fraction(0))


def _recording(policy: Callable[[Observation], np.ndarray], log: list) -> Callable[[Observation], np.ndarray]:
    def wrapped(obs: Observation) -> np.ndarray:
        action = policy(obs)
        log.append((obs, action))
        return action
    return wrapped


def test_binary_search_mean_n64() -> None:
    env = SearchEnv(EnvConfig(), _batch([list(range(0, 256, 4))] * 64, list(range(64))))
    rollout(env, binary_search)
    assert env.success.all() and env.queries_used.sum() == 328
    assert env.queries_used.mean() == 5.125


def test_binary_search_every_n() -> None:
    cfg = EnvConfig(n_min=1)  #default cap: a failure would show as success == False
    rows, tidx = [], []
    for n in range(1, 65):
        for j in range(n):
            rows.append(list(range(0, 4 * n, 4)))
            tidx.append(j)
    env = SearchEnv(cfg, _batch(rows, tidx))
    rollout(env, binary_search)
    assert env.success.all()
    total = [0]
    for m in range(1, 65):
        total.append(m + total[(m - 1) // 2] + total[m // 2])  #S(m) = m + S(floor((m-1)/2)) + S(ceil((m-1)/2))
    sizes, q = np.array([len(r) for r in rows]), env.queries_used
    for n in range(1, 65):
        assert q[sizes == n].sum() == total[n]
        assert q[sizes == n].max() == n.bit_length() <= cfg.query_cap(n)  #bit_length == ceil(log2(n + 1))


@pytest.mark.parametrize("n, target_idx, expected", [
    (64, 0, [31, 15, 7, 3, 1, 0]),
    (64, 63, [31, 47, 55, 59, 61, 62, 63]),
    (4, 3, [1, 2, 3]),
    (2, 1, [0, 1]),
])
def test_binary_search_floor_midpoint(n: int, target_idx: int, expected: list[int]) -> None:
    env = SearchEnv(EnvConfig(n_min=1), _batch([list(range(0, 4 * n, 4))], [target_idx]))
    rollout(env, binary_search)  #costs are symmetric under the upper midpoint, so only the sequence pins it
    assert env.observe().queries[0, :len(expected)].tolist() == expected and env.queries_used[0] == len(expected)


@pytest.mark.parametrize("name", POLICY_NAMES)
@pytest.mark.parametrize("setting", list(SETTINGS))
def test_teachers_terminate_on_target(setting: str, name: str) -> None:
    cfg = SETTINGS[setting]
    batch = sample_batch(cfg, 5000, np.random.default_rng(list(SETTINGS).index(setting)))
    env = SearchEnv(cfg, batch)
    log: list = []
    rollout(env, _recording(make_policy(name, cfg), log))
    assert env.success.all() and np.all(env.queries_used <= batch.n)
    for obs, action in log:  #every query inside the comparison bracket: each one hits or removes an index
        lo, hi = bracket(obs)
        rows = _open_rows(obs)
        assert np.all((lo[rows] <= action[rows]) & (action[rows] <= hi[rows]))


def test_probe_extreme_hits_first_on_trap_s1() -> None:
    cfg = SETTINGS["trap_s1"]
    env = SearchEnv(cfg, sample_batch(cfg, 5000, np.random.default_rng(10)))
    rollout(env, make_policy("probe_extreme_bisect", cfg))
    assert env.success.all() and np.all(env.queries_used == 1)


@pytest.mark.parametrize("setting", ["uniform", "mixture_trap", "clustered"])
def test_interpolate_once_is_binary_after_first_query(setting: str) -> None:
    cfg = SETTINGS[setting]
    env = SearchEnv(cfg, sample_batch(cfg, 2000, np.random.default_rng(20)))
    log: list = []
    rollout(env, _recording(make_policy("interpolate_once", cfg), log))
    states = [obs for obs, _ in log] + [obs for _, obs in _histories(cfg, 200, 8, 21)]
    later = 0
    for obs in states:
        rows = _open_rows(obs)
        first = obs.steps[rows] == 0
        act = interpolate_once(obs, V)[rows]
        assert np.array_equal(act[first], interpolation_search(obs, V)[rows][first])
        assert np.array_equal(act[~first], binary_search(obs)[rows][~first])
        later += int((~first).sum())
    assert later > 1000


@pytest.mark.parametrize("rows, target_idx, queries, expected_bracket, expected", [
    ([list(range(3, 256, 4))], [31], [], (0, 63, -1, 256), 31),  #m=64, L=256, t=128: 63 * 127 / 255 = 31.38
    ([[3, 17, 40, 41, 90, 120, 200, 250]], [5], [], (0, 7, -1, 256), 3),  #7 * 120 / 255 = 3.29
    ([[3, 17, 40, 41, 90, 120, 200, 250]], [5], [3], (4, 7, 41, 256), 5),  #m=4, L=214, t=79: 3 * 78 / 213 = 1.10
    ([[10, 11, 12, 15, 16]], [2], [0, 4], (1, 3, 10, 16), 2),  #m=3, L=5, t=2: exactly 1/2, rounds up to lo + 1
    ([[4, 5, 6, 7, 8]], [2], [0, 4], (1, 3, 4, 8), 2),  #dense bracket L == m: hits
])
def test_interpolation_hand_cases(rows: list[list[int]], target_idx: list[int], queries: list[int],
                                  expected_bracket: tuple[int, int, int, int], expected: int) -> None:
    obs = _observe_after(EnvConfig(feedback="val"), rows, target_idx, queries)
    lo, hi = bracket(obs)
    a, b = value_bounds(obs, V)
    assert (int(lo[0]), int(hi[0]), int(a[0]), int(b[0])) == expected_bracket
    lo_, hi_, a_, b_ = expected_bracket
    m, size, t = hi_ - lo_ + 1, b_ - a_ - 1, int(obs.target[0]) - a_
    assert lo_ + math.floor(_hypergeom_mean(m, size, t) + Fraction(1, 2)) == expected
    assert interpolation_search(obs, V)[0] == expected


def test_interpolation_matches_hypergeometric_mean() -> None:
    checked = 0
    for k, setting in enumerate(["uniform", "mixture_trap", "clustered"]):
        for batch, obs in _histories(SETTINGS[setting], 40, 8, 30 + k):
            act = interpolation_search(obs, V)
            for i in _open_rows(obs):
                lo, hi, a, b = _brute(batch, obs, int(i), V)
                m, size, t = hi - lo + 1, b - a - 1, int(batch.target[i]) - a
                assert act[i] == lo + math.floor(_hypergeom_mean(m, size, t) + Fraction(1, 2))
                checked += 1
    assert checked >= 300


def test_interpolation_hits_dense() -> None:
    cfg = EnvConfig(v=16, n_min=16, n_max=16, feedback="val")
    obs = _observe_after(cfg, [list(range(16))] * 16, list(range(16)), [])
    assert np.array_equal(interpolation_search(obs, 16), np.arange(16))
    dense = 0
    for setting, small in SMALL.items():
        for batch, obs in _histories(small, 200, 4, 40):
            lo, hi = bracket(obs)
            a, b = value_bounds(obs, small.v)
            rows = _open_rows(obs)
            rows = rows[(hi - lo + 1)[rows] == (b - a - 1)[rows]]  #every integer between a and b is in the array
            assert np.array_equal(interpolation_search(obs, small.v)[rows], batch.target_idx[rows])
            dense += len(rows)
    assert dense > 20


@pytest.mark.parametrize("feedback", ["cmp", "val"])
@pytest.mark.parametrize("setting", ["uniform", "mixture_trap", "clustered"])
def test_bracket_matches_brute_force(feedback: str, setting: str) -> None:
    cfg = dataclasses.replace(SETTINGS[setting], feedback=feedback)
    checked = 0
    for batch, obs in _histories(cfg, 60, 10, 50):
        lo, hi = bracket(obs)
        if feedback == "val":
            a, b = value_bounds(obs, V)
            flo, fhi = value_feasible_bracket(obs, V)
        for i in _open_rows(obs):
            want = _brute(batch, obs, int(i), V)
            assert (lo[i], hi[i]) == want[:2]
            if feedback == "val":
                assert (a[i], b[i]) == want[2:]
                assert lo[i] <= flo[i] <= batch.target_idx[i] <= fhi[i] <= hi[i]
            checked += 1
    assert checked > 300


@pytest.mark.parametrize("setting", list(SMALL))
def test_value_feasible_bracket_enumeration(setting: str) -> None:
    cfg = SMALL[setting]
    arrays = {n: list(itertools.combinations(range(cfg.v), n)) for n in range(1, cfg.n_max + 1)}
    checked, narrower = 0, 0
    for batch, obs in _histories(cfg, 80, 4, 60):
        lo, hi = bracket(obs)
        flo, fhi = value_feasible_bracket(obs, cfg.v)
        for i in _open_rows(obs):
            n, t = int(batch.n[i]), int(batch.target[i])
            seen = list(zip(obs.queries[i, :obs.steps[i]].tolist(), obs.feedback[i, :obs.steps[i]].tolist()))
            positions = {s.index(t) for s in arrays[n] if t in s and all(s[q] == f for q, f in seen)}
            assert positions == set(range(int(flo[i]), int(fhi[i]) + 1))
            checked += 1
            narrower += int(fhi[i] - flo[i] < hi[i] - lo[i])
    assert checked > 100 and narrower > 20


def test_teachers_stay_in_range() -> None:
    checked = 0
    for k, setting in enumerate(["uniform", "mixture_trap", "clustered"]):
        cfg = SETTINGS[setting]
        policies = {name: make_policy(name, cfg) for name in POLICY_NAMES}
        for _, obs in _histories(cfg, 100, 10, 70 + k):
            rows = _open_rows(obs)
            lo, hi = (x[rows] for x in bracket(obs))
            flo, fhi = (x[rows] for x in value_feasible_bracket(obs, V))
            first = obs.steps[rows] == 0
            for name, policy in policies.items():
                act = policy(obs)[rows]
                assert np.all((lo <= act) & (act <= hi)), name
                feasible = (flo <= act) & (act <= fhi)
                if name in ALWAYS_FEASIBLE:
                    assert np.all(feasible), name
                elif name != "binary":
                    assert np.all(feasible[first]), name  #binary after the first query need not be value-feasible
            checked += len(rows)
    assert checked > 1000


@pytest.mark.parametrize("n, t, expected", [
    (64, 0, 0), (64, 10, 10), (64, 127, 63), (64, 128, 0), (64, 255, 63), (256, 100, 100), (256, 200, 200),
    (1, 0, 0), (1, 255, 0),
])
def test_extreme_rank_hand(n: int, t: int, expected: int) -> None:
    assert extreme_rank(np.array([n]), np.array([t]), V)[0] == expected


def test_extreme_rank_matches_trap_window() -> None:
    rng = np.random.default_rng(80)
    n = rng.integers(1, V + 1, size=20_000)
    t = rng.integers(0, V, size=20_000)
    j0 = rng.integers(0, n)
    lo, hi = trap_rank_window(n, t, j0, V, 1.0)
    k = extreme_rank(n, t, V)
    assert np.array_equal(lo, k) and np.array_equal(hi, k)


def test_probe_extreme_first_query_then_continuation() -> None:
    later = 0
    for _, obs in _histories(SETTINGS["mixture_trap"], 200, 8, 90):
        rows = _open_rows(obs)
        first = obs.steps[rows] == 0
        k = extreme_rank(obs.n, obs.target, V)[rows]
        for then, rest in (("bisect", binary_search(obs)), ("interpolate", interpolation_search(obs, V))):
            act = probe_extreme(obs, V, then)[rows]
            assert np.array_equal(act[first], k[first])
            assert np.array_equal(act[~first], rest[rows][~first])
        later += int((~first).sum())
    assert later > 500
    obs = _observe_after(EnvConfig(feedback="val"), [[1, 2, 3]], [1], [])
    with pytest.raises(ValueError):
        probe_extreme(obs, V, "interpolation")


@pytest.mark.parametrize("setting", ["uniform", "mixture_trap", "clustered"])
def test_binary_ignores_values(setting: str) -> None:
    val = SETTINGS[setting]
    cmp = dataclasses.replace(val, feedback="cmp")
    batch = sample_batch(val, 2000, np.random.default_rng(100))
    envs = [SearchEnv(cfg, batch) for cfg in (cmp, val)]
    for env in envs:
        rollout(env, binary_search)
    obs_cmp, obs_val = (env.observe() for env in envs)
    assert np.array_equal(obs_cmp.queries, obs_val.queries) and np.array_equal(obs_cmp.steps, obs_val.steps)


def test_comparison_codes_hand() -> None:
    rows, tidx, queries = [[0, 5, 9, 14]], [2], [0, 3, 2]
    want = np.array([[LESS, GREATER, EQUAL] + [PAD] * 61])  #k_max = 64 under OPEN; unused slots stay PAD
    for feedback in ("cmp", "val"):
        obs = _observe_after(EnvConfig(feedback=feedback, **OPEN), rows, tidx, queries)
        assert np.array_equal(comparison_codes(obs), want)
    obs = _observe_after(EnvConfig(feedback="val", **OPEN), rows, tidx, [0])
    assert np.array_equal(comparison_codes(obs)[0, :2], [LESS, PAD])  #A[0] = 0: a value, not padding


def test_comparison_codes_val_equals_cmp_feedback() -> None:
    val = SETTINGS["mixture_trap"]
    cmp = dataclasses.replace(val, feedback="cmp")
    rng = np.random.default_rng(110)
    batch = sample_batch(val, 1000, rng)
    env_cmp, env_val = SearchEnv(cmp, batch), SearchEnv(val, batch)
    for _ in range(12):
        q = rng.integers(0, batch.n)
        env_cmp.step(q)
        env_val.step(q)
        obs_cmp, obs_val = env_cmp.observe(), env_val.observe()
        assert np.array_equal(comparison_codes(obs_val), obs_cmp.feedback)
        assert np.array_equal(comparison_codes(obs_cmp), obs_cmp.feedback)


def test_make_policy_dispatch() -> None:
    cfg = SETTINGS["mixture_trap"]
    direct = {
        "binary": binary_search,
        "interpolation": lambda obs: interpolation_search(obs, V),
        "interpolate_once": lambda obs: interpolate_once(obs, V),
        "vf_midpoint": lambda obs: value_feasible_midpoint(obs, V),
        "guarded_interpolation": lambda obs: guarded_interpolation(obs, V, 0.25),
        "interpolate_k_vf": lambda obs: interpolate_k_then_vf_midpoint(obs, V, 1),
        "adaptive_switch": lambda obs: adaptive_switch(obs, V, 0.5),
        "probe_extreme_bisect": lambda obs: probe_extreme(obs, V, "bisect"),
        "probe_extreme_interpolate": lambda obs: probe_extreme(obs, V, "interpolate"),
    }
    assert set(direct) == set(POLICY_NAMES) and VALUE_BASED == set(POLICY_NAMES) - {"binary"}
    assert ALWAYS_FEASIBLE <= VALUE_BASED
    for _, obs in _histories(cfg, 100, 4, 120):
        for name in POLICY_NAMES:
            assert np.array_equal(make_policy(name, cfg)(obs), direct[name](obs)), name
        assert np.array_equal(make_policy("interpolate_k_vf", cfg, k=2)(obs), interpolate_k_then_vf_midpoint(obs, V, 2))
    with pytest.raises(ValueError):
        make_policy("binary", cfg, k=2)


def test_value_based_rejected_under_cmp() -> None:
    cmp = EnvConfig()
    make_policy("binary", cmp)
    for name in VALUE_BASED:
        with pytest.raises(ValueError):
            make_policy(name, cmp)
    with pytest.raises(ValueError):
        make_policy("ternary", SETTINGS["uniform"])
    obs = _observe_after(cmp, [[1, 2, 3]], [1], [0])
    assert obs.target is None
    for fn in (lambda o: interpolation_search(o, V), lambda o: interpolate_once(o, V),
               lambda o: probe_extreme(o, V, "bisect"), lambda o: value_bounds(o, V),
               lambda o: value_feasible_bracket(o, V), lambda o: value_feasible_midpoint(o, V),
               lambda o: guarded_interpolation(o, V, 0.25), lambda o: interpolate_k_then_vf_midpoint(o, V, 1),
               lambda o: adaptive_switch(o, V, 0.5)):
        with pytest.raises(ValueError):
            fn(obs)


def test_vf_midpoint_hand_and_equals_binary_when_unnarrowed() -> None:
    obs = _observe_after(EnvConfig(feedback="val"), [[10, 11, 12, 15, 16]], [2], [0, 4])
    assert [int(x[0]) for x in bracket(obs)] == [1, 3] and [int(x[0]) for x in value_feasible_bracket(obs, V)] == [1, 2]
    assert value_feasible_midpoint(obs, V)[0] == 1 and binary_search(obs)[0] == 2
    obs = _observe_after(EnvConfig(feedback="val"), [[3, 17, 40, 41, 90, 120, 200, 250]], [5], [3])
    assert [int(x[0]) for x in bracket(obs)] == [4, 7] and [int(x[0]) for x in value_feasible_bracket(obs, V)] == [4, 7]
    assert value_feasible_midpoint(obs, V)[0] == 5 and binary_search(obs)[0] == 5
    checked, differ = 0, 0
    for k, setting in enumerate(["uniform", "mixture_trap", "clustered"]):
        for _, obs in _histories(SETTINGS[setting], 200, 8, 130 + k):
            rows = _open_rows(obs)
            lo, hi = (x[rows] for x in bracket(obs))
            vlo, vhi = (x[rows] for x in value_feasible_bracket(obs, V))
            act, bin_ = value_feasible_midpoint(obs, V)[rows], binary_search(obs)[rows]
            assert np.all((vlo <= act) & (act <= vhi))
            same = (vlo == lo) & (vhi == hi)
            assert np.array_equal(act[same], bin_[same])
            differ += int((act != bin_).sum())
            checked += len(rows)
    assert checked >= 1000 and differ > 0


def test_guarded_interpolation_band() -> None:
    states = []
    for k, setting in enumerate(["uniform", "mixture_trap", "clustered"]):
        states += [(SETTINGS[setting], batch, obs) for batch, obs in _histories(SETTINGS[setting], 200, 8, 140 + k)]
    for g in (0.0, 0.25, 0.4):
        clamped, less, greater = 0, 0, 0
        for cfg, batch, obs in states:
            rows = _open_rows(obs)
            vlo, vhi = (x[rows] for x in value_feasible_bracket(obs, V))
            w = vhi - vlo + 1
            pad = np.array([math.floor(Fraction(str(g)) * int(x)) for x in w], dtype=np.int64)  #exact floor(g w)
            assert np.all(vlo + pad <= vhi - pad)
            action = guarded_interpolation(obs, V, g)
            act, interp = action[rows], interpolation_search(obs, V)[rows]
            assert np.all((vlo + pad <= act) & (act <= vhi - pad))
            inside = (vlo + pad <= interp) & (interp <= vhi - pad)
            assert np.array_equal(act[inside], interp[inside])
            if g == 0.0:
                assert np.array_equal(act, interp)
            clamped += int((~inside).sum())
            after = _extend(cfg, batch, obs, action)  #one more guarded query: each outcome removes >= floor(g w) + 1
            miss = ~after.done[rows]
            assert np.all((w - _width(after)[rows])[miss] >= pad[miss] + 1)
            code = comparison_codes(after)[rows, obs.steps[rows]][miss]
            less, greater = less + int((code == LESS).sum()), greater + int((code == GREATER).sum())
        assert less > 300 and greater > 300
        if g == 0.25:
            assert clamped > 0
    for setting in ("uniform", "mixture_trap"):
        cfg = SETTINGS[setting]
        env = SearchEnv(cfg, sample_batch(cfg, 5000, np.random.default_rng(150)))
        log: list = []
        rollout(env, _recording(make_policy("guarded_interpolation", cfg), log))
        assert env.success.all()
        states_ = [obs for obs, _ in log] + [env.observe()]
        for before, after in zip(states_, states_[1:]):
            rows = np.nonzero(~before.done & ~after.done)[0]
            w = _width(before)[rows]
            assert np.all(w - _width(after)[rows] >= w // 4 + 1)  #g = 0.25
    obs = _observe_after(EnvConfig(feedback="val"), [[1, 2, 3]], [1], [])
    with pytest.raises(ValueError):
        guarded_interpolation(obs, V, 0.5)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_interpolate_k_then_vf_midpoint(k: int) -> None:
    states = []
    for j, setting in enumerate(["uniform", "mixture_trap"]):
        cfg = SETTINGS[setting]
        env = SearchEnv(cfg, sample_batch(cfg, 2000, np.random.default_rng(160 + j)))
        log: list = []
        rollout(env, _recording(make_policy("interpolate_k_vf", cfg, k=k), log))
        assert env.success.all()
        states += [obs for obs, _ in log] + [obs for _, obs in _histories(cfg, 200, 8, 170 + j)]
    early, late, differ = 0, 0, 0
    for obs in states:
        rows = _open_rows(obs)
        first = obs.steps[rows] < k
        act = interpolate_k_then_vf_midpoint(obs, V, k)[rows]
        vfm = value_feasible_midpoint(obs, V)[rows]
        assert np.array_equal(act[first], interpolation_search(obs, V)[rows][first])
        assert np.array_equal(act[~first], vfm[~first])
        if k == 1:
            expected = (obs.steps[rows] >= 1) & (vfm != binary_search(obs)[rows])
            assert np.array_equal(act != interpolate_once(obs, V)[rows], expected)
            differ += int(expected.sum())
        early, late = early + int(first.sum()), late + int((~first).sum())
    assert early > 500 and late > 500 and (k != 1 or differ > 0)
    obs = _observe_after(EnvConfig(feedback="val"), [[1, 2, 3]], [1], [])
    with pytest.raises(ValueError):
        interpolate_k_then_vf_midpoint(obs, V, 0)


def test_value_feasible_widths_matches_prefix_replay() -> None:
    checked, narrow_start = 0, 0
    for k, setting in enumerate(["uniform", "mixture_trap"]):
        cfg = SETTINGS[setting]
        for batch, obs in _histories(cfg, 30, 6, 180 + k):
            widths = value_feasible_widths(obs, V)
            assert widths.shape == (len(obs.n), obs.queries.shape[1] + 1)
            for i in _open_rows(obs):
                n, steps = int(batch.n[i]), int(obs.steps[i])
                row, qs = batch.values[i, :n].tolist(), obs.queries[i, :steps].tolist()
                for s in range(steps + 1):  #s = 0 is the fresh observation: T alone can narrow the range below [0, N)
                    assert widths[i, s] == _width(_observe_after(cfg, [row], [int(batch.target_idx[i])], qs[:s]))[0]
                assert np.all(widths[i, steps:] == widths[i, steps])
                narrow_start += int(widths[i, 0] < n)
                checked += 1
    assert checked > 200 and narrow_start > 0


def _switched_by_replay(cfg: EnvConfig, batch: EpisodeBatch, obs: Observation, i: int, theta: Fraction) -> bool:
    n, steps = int(batch.n[i]), int(obs.steps[i])
    row, qs, tidx = batch.values[i, :n].tolist(), obs.queries[i, :steps].tolist(), [int(batch.target_idx[i])]
    w = [int(_width(_observe_after(cfg, [row], tidx, qs[:s]))[0]) for s in range(steps + 1)]
    return any(Fraction(w[s] - w[s + 1], w[s]) < theta for s in range(steps))


def test_adaptive_switch() -> None:
    cfg = EnvConfig(feedback="val")
    row = [[12, 49, 102, 108, 110, 119, 126, 240]]
    for queries, expected in (([4], "interp"), ([0], "vfm"), ([0, 4], "vfm")):  #[4]: 8 -> 2; [0]: 8 -> 7; then 7 -> 2
        obs = _observe_after(cfg, row, [3], queries)
        interp, vfm = int(interpolation_search(obs, V)[0]), int(value_feasible_midpoint(obs, V)[0])
        assert interp != vfm
        assert adaptive_switch(obs, V, 0.5)[0] == (interp if expected == "interp" else vfm)
    switched_rows, kept_rows = 0, 0
    for k, setting in enumerate(["uniform", "mixture_trap", "clustered"]):
        cfg = SETTINGS[setting]
        for batch, obs in _histories(cfg, 40, 6, 190 + k):
            act, interp, vfm = adaptive_switch(obs, V, 0.5), interpolation_search(obs, V), value_feasible_midpoint(obs, V)
            for i in _open_rows(obs):
                switched = _switched_by_replay(cfg, batch, obs, int(i), Fraction(1, 2))
                assert act[i] == (vfm[i] if switched else interp[i])
                if interp[i] != vfm[i]:
                    switched_rows += int(switched)
                    kept_rows += int(not switched and obs.steps[i] >= 1)
    assert switched_rows > 0 and kept_rows > 0
    for k, setting in enumerate(["uniform", "mixture_trap", "clustered"]):
        cfg = SETTINGS[setting]
        env = SearchEnv(cfg, sample_batch(cfg, 5000, np.random.default_rng(200 + k)))
        log: list = []
        rollout(env, _recording(make_policy("adaptive_switch", cfg), log))
        assert env.success.all()
        if setting == "uniform":
            ever = np.zeros(5000, dtype=bool)  #switch state that governed an action; the hitting query ends the episode
            for obs, _ in log:
                rows = _open_rows(obs)
                w = value_feasible_widths(obs, V)
                taken = np.arange(obs.queries.shape[1])[None, :] < obs.steps[:, None]
                ever[rows] |= np.any(taken & ((w[:, :-1] - w[:, 1:]) * 2 < w[:, :-1]), axis=1)[rows]
            assert ever.mean() < 0.5
    obs = _observe_after(EnvConfig(feedback="val"), [[1, 2, 3]], [1], [])
    for theta in (0.0, 1.5):
        with pytest.raises(ValueError):
            adaptive_switch(obs, V, theta)
