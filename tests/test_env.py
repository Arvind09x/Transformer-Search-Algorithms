import dataclasses
import itertools
import math
from collections import Counter

import numpy as np
import pytest

import env as E
from config import EnvConfig
from env import EQUAL, GREATER, LESS, PAD, EpisodeBatch, Observation, SearchEnv, rollout, sample_batch

SETTINGS = {
    "uniform": EnvConfig(),
    "adversarial_trap": EnvConfig(distribution="adversarial", adversarial="trap"),
    "mixture_trap": EnvConfig(distribution="mixture", adversarial="trap", p=0.3),
    "adversarial_clustered": EnvConfig(distribution="adversarial", adversarial="clustered"),
    "mixture_clustered": EnvConfig(distribution="mixture", adversarial="clustered", p=0.3),
}


def _chi2_check(counts: Counter, law: dict, m: int) -> None:
    assert math.isclose(sum(law.values()), 1.0, abs_tol=1e-12)
    unexpected = {c: k for c, k in counts.items() if law.get(c, 0.0) == 0.0}
    assert not unexpected, f"zero-probability cells observed: {list(unexpected)[:5]}"
    support = {c: p for c, p in law.items() if p > 0.0}
    stat = sum((counts.get(c, 0) - m * p) ** 2 / (m * p) for c, p in support.items())
    df = len(support) - 1
    assert stat < df + 6.0 * math.sqrt(2.0 * df), f"chi2={stat:.1f}, df={df}"  #about p < 1e-5


def _cells(b: EpisodeBatch) -> Counter:
    n = int(b.n[0])
    return Counter(zip(map(tuple, b.values[:, :n].tolist()), b.target_idx.tolist()))


def _sequential_set_prob(s: tuple, w: list[float]) -> float:
    total = 0.0
    for perm in itertools.permutations(s):  #successive sampling: sum over every order of drawing the set
        rem, p = sum(w), 1.0
        for x in perm:
            p *= w[x] / rem
            rem -= w[x]
        total += p
    return total


def _gauss_weights(centres: tuple, sigma: float, v: int) -> list[float]:
    return [sum(math.exp(-((x - c) ** 2) / (2.0 * sigma ** 2)) for c in centres) for x in range(v)]


def _round_away(x: float) -> int:
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def _law_uniform(v: int, n: int) -> dict:
    sets = list(itertools.combinations(range(v), n))
    return {(s, j): 1.0 / (len(sets) * n) for s in sets for j in range(n)}


def _law_clustered(v: int, n: int, k: int, sigma: float) -> dict:
    law: dict = {}
    for centres in itertools.product(range(v), repeat=k):
        w = _gauss_weights(centres, sigma, v)
        for s in itertools.combinations(range(v), n):
            ps = _sequential_set_prob(s, w) / (v ** k * n)
            for j in range(n):
                law[(s, j)] = law.get((s, j), 0.0) + ps
    return law


def _law_trap(v: int, n: int, strength: float) -> dict:
    law: dict = {}
    sets = list(itertools.combinations(range(v), n))
    for a0 in sets:
        for j0 in range(n):
            t = a0[j0]
            r_lo, r_hi = max(0, n - v + t), min(n - 1, t)
            k_ext = r_hi if t < v / 2 else r_lo
            edge = j0 + _round_away(strength * (k_ext - j0))
            lo, hi = min(edge, k_ext), max(edge, k_ext)
            for kp in range(lo, hi + 1):
                below = list(itertools.combinations(range(t), kp))
                above = list(itertools.combinations(range(t + 1, v), n - 1 - kp))
                p = 1.0 / (len(sets) * n * (hi - lo + 1) * len(below) * len(above))
                for bs in below:
                    for us in above:
                        key = (tuple(sorted(bs + (t,) + us)), kp)
                        law[key] = law.get(key, 0.0) + p
    return law


def _batch(rows: list[list[int]], target_idx: list[int], v: int = 256) -> EpisodeBatch:
    n = np.array([len(r) for r in rows])
    values = np.full((len(rows), int(n.max())), v, dtype=np.int64)
    for i, r in enumerate(rows):
        values[i, :len(r)] = r
    t = np.asarray(target_idx, dtype=np.int64)
    return EpisodeBatch(n=n, values=values, target_idx=t, target=values[np.arange(len(rows)), t],
                        adversarial=np.zeros(len(rows), dtype=bool))


def _random_policy(seed: int):
    rng = np.random.default_rng(seed)
    return lambda obs: rng.integers(0, obs.n)




def test_uniform_law() -> None:
    cfg, m = EnvConfig(v=6, n_min=3, n_max=3), 60_000
    _chi2_check(_cells(sample_batch(cfg, m, np.random.default_rng(0))), _law_uniform(6, 3), m)


def test_clustered_values_law_fixed_centres() -> None:
    v, n, sigma, m = 8, 3, 1.5, 100_000
    centres = np.tile([1, 6], (m, 1))
    vals = E.sample_clustered_values(np.full(m, n), centres, sigma, v, n, np.random.default_rng(0))
    w = _gauss_weights((1, 6), sigma, v)
    law = {s: _sequential_set_prob(s, w) for s in itertools.combinations(range(v), n)}
    _chi2_check(Counter(map(tuple, vals.tolist())), law, m)


def test_clustered_law_end_to_end() -> None:
    cfg = EnvConfig(v=8, n_min=3, n_max=3, distribution="adversarial", adversarial="clustered", clustered_sigma=1.5)
    m = 168_000
    _chi2_check(_cells(sample_batch(cfg, m, np.random.default_rng(0))), _law_clustered(8, 3, 2, 1.5), m)


@pytest.mark.parametrize("strength", [0.5, 0.75])  #x = +-0.5 and -1.5 occur at N = 3: separates rounding rules
def test_trap_law(strength: float) -> None:
    cfg = EnvConfig(v=8, n_min=3, n_max=3, distribution="adversarial", adversarial="trap", trap_strength=strength)
    m = 100_000
    _chi2_check(_cells(sample_batch(cfg, m, np.random.default_rng(0))), _law_trap(8, 3, strength), m)


def test_n_law() -> None:
    cfg, m = EnvConfig(v=64, n_min=4, n_max=64), 61_000
    counts = Counter(sample_batch(cfg, m, np.random.default_rng(0)).n.tolist())
    _chi2_check(counts, {k: 1.0 / 61 for k in range(4, 65)}, m)


def test_mixture_flag_rate() -> None:
    cfg, m, p = EnvConfig(v=16, n_min=4, n_max=8, distribution="mixture", adversarial="trap", p=0.1), 100_000, 0.1
    rate = sample_batch(cfg, m, np.random.default_rng(0)).adversarial.mean()
    assert abs(rate - p) < 5.0 * math.sqrt(p * (1.0 - p) / m)




@pytest.mark.parametrize("name", list(SETTINGS))
def test_rows_valid(name: str) -> None:
    cfg = SETTINGS[name]
    b = sample_batch(cfg, 2000, np.random.default_rng(1))
    assert np.all((b.n >= cfg.n_min) & (b.n <= cfg.n_max))
    assert b.values.shape == (2000, cfg.n_max)
    for r in range(2000):
        row = b.values[r, :b.n[r]]
        assert np.all(np.diff(row) > 0) and row[0] >= 0 and row[-1] < cfg.v
        assert np.all(b.values[r, b.n[r]:] == cfg.v)
    assert np.array_equal(b.values[np.arange(2000), b.target_idx], b.target)
    assert np.all((b.target_idx >= 0) & (b.target_idx < b.n))


def test_trap_rank_in_window_and_feasible() -> None:
    cfg = EnvConfig(n_min=64, distribution="adversarial", adversarial="trap")
    u = sample_batch(dataclasses.replace(cfg, distribution="uniform"), 20_000, np.random.default_rng(2))
    t = sample_batch(cfg, 20_000, np.random.default_rng(2))
    assert np.array_equal(u.target, t.target)
    lo, hi = E.trap_rank_window(u.n, u.target, u.target_idx, cfg.v, cfg.trap_strength)
    r_lo, r_hi = np.maximum(0, u.n - cfg.v + u.target), np.minimum(u.n - 1, u.target)
    assert np.all((r_lo <= lo) & (lo <= t.target_idx) & (t.target_idx <= hi) & (hi <= r_hi))


def test_round_half_away() -> None:
    x = np.array([-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, -0.49, 0.49])
    assert E.round_half_away(x).tolist() == [-3, -2, -1, 0, 1, 2, 3, 0, 0]


@pytest.mark.parametrize("n, t, j0, s, expected", [
    (64, 10, 2, 0.75, (8, 10)),  #T < V/2: k_ext = r_hi = 10, edge = 2 + 6
    (64, 200, 50, 0.75, (8, 18)),  #T >= V/2: k_ext = r_lo = 8, edge = 50 - 32 (-31.5 rounds away)
    (64, 100, 30, 1.0, (63, 63)),  #s = 1: point mass at k_ext
    (64, 100, 30, 0.0, (30, 63)),  #s = 0: widest window [j0, k_ext]
    (64, 0, 0, 0.75, (0, 0)),  #T = 0
    (64, 255, 63, 0.75, (63, 63)),  #T = V - 1
    (256, 77, 77, 0.75, (77, 77)),  #N = V: r_lo = r_hi = T
])
def test_trap_rank_window_values(n: int, t: int, j0: int, s: float, expected: tuple[int, int]) -> None:
    lo, hi = E.trap_rank_window(np.array([n]), np.array([t]), np.array([j0]), 256, s)
    assert (int(lo[0]), int(hi[0])) == expected


def test_trap_edge_cases() -> None:
    rng = np.random.default_rng(3)
    vals = E.sample_trap_values(np.array([4, 4]), np.array([0, 15]), np.array([0, 3]), 16, 4, rng)
    assert vals[0, 0] == 0 and vals[1, 3] == 15
    cfg = EnvConfig(v=16, n_min=16, n_max=16, distribution="adversarial", adversarial="trap")
    b = sample_batch(cfg, 500, np.random.default_rng(4))
    assert np.array_equal(b.values, np.tile(np.arange(16), (500, 1)))
    assert np.array_equal(b.target_idx, b.target)


def test_common_random_numbers() -> None:
    def draw(cfg: EnvConfig) -> EpisodeBatch:
        return sample_batch(cfg, 3000, np.random.default_rng(5))
    uni = draw(SETTINGS["uniform"])
    for name in ("adversarial_trap", "mixture_trap"):
        b = draw(SETTINGS[name])
        assert np.array_equal(b.n, uni.n) and np.array_equal(b.target, uni.target)
    assert np.array_equal(draw(SETTINGS["adversarial_clustered"]).n, uni.n)
    for comp in ("trap", "clustered"):
        adv, mix = draw(SETTINGS[f"adversarial_{comp}"]), draw(SETTINGS[f"mixture_{comp}"])
        f = mix.adversarial
        assert 0 < f.sum() < len(f)
        assert np.array_equal(mix.values, np.where(f[:, None], adv.values, uni.values))
        assert np.array_equal(mix.target_idx, np.where(f, adv.target_idx, uni.target_idx))


def test_nested_flags() -> None:
    base = EnvConfig(v=16, n_min=4, n_max=8, distribution="mixture", adversarial="trap")
    def draw(p: float) -> EpisodeBatch:
        return sample_batch(dataclasses.replace(base, p=p), 5000, np.random.default_rng(6))
    lo, hi = draw(0.05), draw(0.25)
    assert np.all(hi.adversarial[lo.adversarial]) and hi.adversarial.sum() > lo.adversarial.sum()
    assert np.array_equal(lo.values[lo.adversarial], hi.values[lo.adversarial])
    assert not draw(0.0).adversarial.any() and draw(1.0).adversarial.all()


def test_determinism() -> None:
    cfg = SETTINGS["mixture_trap"]
    a, b = sample_batch(cfg, 500, np.random.default_rng(7)), sample_batch(cfg, 500, np.random.default_rng(7))
    for f in dataclasses.fields(EpisodeBatch):
        assert np.array_equal(getattr(a, f.name), getattr(b, f.name))
    rng = np.random.default_rng(7)
    assert not np.array_equal(sample_batch(cfg, 500, rng).values, sample_batch(cfg, 500, rng).values)




def test_cmp_hides_target_relabelling() -> None:
    def trace(cfg: EnvConfig, b: EpisodeBatch) -> list[Observation]:
        env, seen, pol = SearchEnv(cfg, b), [], _random_policy(8)
        def record(obs: Observation) -> np.ndarray:
            seen.append(obs)
            return pol(obs)
        rollout(env, record)
        return seen + [env.observe()]
    cfg = EnvConfig(n_min=32, n_max=32)
    b1 = sample_batch(cfg, 400, np.random.default_rng(9))
    b2 = sample_batch(cfg, 400, np.random.default_rng(10))
    relabelled = EpisodeBatch(n=b1.n, values=b2.values, target_idx=b1.target_idx,
                              target=b2.values[np.arange(400), b1.target_idx], adversarial=b1.adversarial)
    assert not np.array_equal(b1.values, relabelled.values)
    t1, t2 = trace(cfg, b1), trace(cfg, relabelled)
    assert len(t1) == len(t2)
    for o1, o2 in zip(t1, t2):
        assert o1.target is None and o2.target is None
        assert set(np.unique(o1.feedback)) <= {PAD, LESS, EQUAL, GREATER}
        for f in dataclasses.fields(Observation):
            if f.name != "target":
                assert np.array_equal(getattr(o1, f.name), getattr(o2, f.name))
    val = dataclasses.replace(cfg, feedback="val")
    v1, v2 = trace(val, b1), trace(val, relabelled)
    assert np.array_equal(v1[0].target, b1.target)
    assert not np.array_equal(v1[-1].feedback, v2[-1].feedback)


def test_feedback_convention_hand() -> None:
    env = SearchEnv(EnvConfig(), _batch([[10, 20, 30, 40]], [2]))
    for idx, code, done in ((0, LESS, False), (3, GREATER, False), (2, EQUAL, True)):
        fb, _, d = env.step(np.array([idx]))
        assert fb[0] == code and d[0] == done
    assert env.success[0] and env.queries_used[0] == 3
    venv = SearchEnv(EnvConfig(feedback="val"), _batch([[10, 20, 30, 40]], [2]))
    assert venv.step(np.array([3]))[0][0] == 40


@pytest.mark.parametrize("feedback", ["cmp", "val"])
def test_feedback_matches_hidden_array(feedback: str) -> None:
    cfg = dataclasses.replace(SETTINGS["mixture_trap"], feedback=feedback)
    b = sample_batch(cfg, 1000, np.random.default_rng(11))
    env = SearchEnv(cfg, b)
    rollout(env, _random_policy(12))
    o = env.observe()
    for r in range(1000):
        k = o.steps[r]
        q = o.queries[r, :k]
        a = b.values[r, q]
        expected = np.sign(a - b.target[r]) + 1 if feedback == "cmp" else a
        assert np.array_equal(o.feedback[r, :k], expected)
        assert np.all(o.queries[r, k:] == PAD) and np.all(o.feedback[r, k:] == PAD)


@pytest.mark.parametrize("query_cost, failure_cost_coef", [(1, 1), (2, 3)])
def test_cost_identity(query_cost: int, failure_cost_coef: int) -> None:
    cfg = EnvConfig(query_cost=query_cost, failure_cost_coef=failure_cost_coef)
    env = SearchEnv(cfg, sample_batch(cfg, 2000, np.random.default_rng(13)))
    rewards = rollout(env, _random_policy(14))
    fail = ~env.success
    assert fail.any() and env.success.any()
    expected = query_cost * env.queries_used + failure_cost_coef * env.cap * fail
    assert np.array_equal(env.cost(), expected)
    assert np.array_equal(-rewards.sum(axis=1), expected)
    assert np.all(env.queries_used[fail] == env.cap[fail])


def _expected_cap(cap_mode: str, n: int) -> int:
    return n if cap_mode == "linear" else 2 * math.ceil(math.log2(n)) + 2


@pytest.mark.parametrize("cap_mode, k_max", [("linear", 64), ("log", 14)])
def test_cap_formula(cap_mode: str, k_max: int) -> None:
    cfg = EnvConfig(n_min=1, cap_mode=cap_mode)
    rows = [list(range(n)) for n in range(1, 65)]
    env = SearchEnv(cfg, _batch(rows, [0] * 64))
    assert env.cap.tolist() == [_expected_cap(cap_mode, n) for n in range(1, 65)]
    assert env.k_max == k_max


@pytest.mark.parametrize("cap_mode", ["linear", "log"])
def test_cap_never_hit_and_hit_on_last_query(cap_mode: str) -> None:
    cfg = EnvConfig(query_cost=2, failure_cost_coef=3, cap_mode=cap_mode)
    ns = list(range(4, 65))
    rows = [list(range(0, 2 * n, 2)) for n in ns]
    miss = SearchEnv(cfg, _batch(rows, [0] * len(ns)))
    rollout(miss, lambda obs: np.ones_like(obs.n))  #repeats index 1, never the target
    cap = np.array([_expected_cap(cap_mode, n) for n in ns])
    assert not miss.success.any() and np.array_equal(miss.queries_used, cap)
    assert np.array_equal(miss.cost(), 2 * cap + 3 * cap)
    late = SearchEnv(cfg, _batch(rows, [0] * len(ns)))
    rollout(late, lambda obs: np.where(obs.steps == cap - 1, 0, 1))
    assert late.success.all() and np.array_equal(late.queries_used, cap)
    assert np.array_equal(late.cost(), 2 * cap)


def test_termination_on_hit() -> None:
    env = SearchEnv(EnvConfig(), _batch([[1, 5, 9, 13], [2, 4, 6, 8]], [1, 3]))
    fb, r, d = env.step(np.array([1, 0]))
    assert d.tolist() == [True, False] and fb[0] == EQUAL
    before = env.observe()
    fb, r, d = env.step(np.array([2, 3]))
    after = env.observe()
    assert fb[0] == PAD and r[0] == 0.0 and after.steps[0] == 1
    assert np.array_equal(before.queries[0], after.queries[0]) and np.array_equal(before.feedback[0], after.feedback[0])
    assert d.all() and env.success.all()


def test_input_validation() -> None:
    def fresh() -> SearchEnv:
        return SearchEnv(EnvConfig(), _batch([[1, 5, 9, 13], [2, 4, 6, 8]], [1, 3]))
    for bad in (np.array([-1, 0]), np.array([4, 0]), np.array([1.0, 0.0]), np.array([1, 0, 0])):
        with pytest.raises(ValueError):
            fresh().step(bad)
    env = fresh()
    env.step(np.array([1, 0]))
    env.step(np.array([99, 3]))  #garbage index on a finished row is ignored
    with pytest.raises(ValueError):
        fresh().cost()


def test_no_bracket_exposed() -> None:
    assert {f.name for f in dataclasses.fields(Observation)} == {"n", "target", "queries", "feedback", "steps", "done"}
    env = SearchEnv(EnvConfig(), _batch([[1, 5, 9, 13]], [1]))
    public = {name for name in dir(env) if not name.startswith("_")}
    assert public == {"cap", "queries_used", "done", "success", "k_max", "observe", "step", "cost"}


def test_isolation() -> None:
    b = _batch([[1, 5, 9, 13]], [1])
    env = SearchEnv(EnvConfig(), b)
    env.step(np.array([0]))
    obs = env.observe()
    obs.queries[:] = 3
    obs.feedback[:] = 0
    obs.steps[:] = 9
    fresh = env.observe()
    assert fresh.queries[0, 0] == 0 and fresh.queries[0, 1] == PAD and fresh.steps[0] == 1
    with pytest.raises(ValueError):
        b.values[0, 0] = 2


def test_rollout() -> None:
    cfg = SETTINGS["mixture_clustered"]
    env = SearchEnv(cfg, sample_batch(cfg, 1000, np.random.default_rng(15)))
    rewards = rollout(env, _random_policy(16))
    assert rewards.shape == (1000, env.k_max) and env.done.all()
    assert np.array_equal(-rewards.sum(axis=1), env.cost())
    cols = np.arange(env.k_max)[None, :]
    assert np.all(rewards[cols >= env.queries_used[:, None]] == 0.0)
    assert np.all(rewards[cols < env.queries_used[:, None]] <= -cfg.query_cost)


def test_binary_search_mean_n64() -> None:
    env = SearchEnv(EnvConfig(), _batch([list(range(0, 256, 4))] * 64, list(range(64))))
    lo, hi = np.zeros(64, dtype=np.int64), np.full(64, 63)
    while not env.done.all():
        mid = (lo + hi) // 2
        fb, _, _ = env.step(mid)
        lo = np.where(fb == LESS, mid + 1, lo)
        hi = np.where(fb == GREATER, mid - 1, hi)
    assert env.success.all() and env.queries_used.sum() == 328  #328 / 64 = 5.125
