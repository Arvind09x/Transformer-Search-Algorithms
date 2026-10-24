import itertools
from fractions import Fraction

import numpy as np
import pytest

from config import EnvConfig
from dp import (TIE_TOL, CmpTable, ValTable, cmp_q_totals, cmp_state, cmp_table, cmp_value, dp_policy,
                enumerate_cmp, enumerate_val_uniform, hypergeom_rank, load_val_table, optimal_mask, q_values,
                save_val_table, val_outcome_given_rank, val_q, val_state, val_uniform_table)
from env import PAD, EpisodeBatch, Observation, SearchEnv, rollout, sample_batch
from teachers import bracket, make_policy, value_feasible_bracket

OPEN = dict(n_min=1, cap_mode="linear")
VAL_SIZES = [(1, 4), (2, 6), (3, 8), (4, 10), (5, 10), (5, 12)]  #(N, V); every target value in [0, V)


@pytest.fixture(scope="module")
def tiny() -> ValTable:
    return val_uniform_table(5, 12)


@pytest.fixture(scope="module")
def mid() -> ValTable:
    return val_uniform_table(16, 64)


def _obs(n: int, target: int | None, history: frozenset[tuple[int, int]]) -> Observation:
    pairs = sorted(history)
    queries = np.full((1, n), PAD, dtype=np.int64)
    feedback = np.full((1, n), PAD, dtype=np.int64)
    for s, (i, f) in enumerate(pairs):
        queries[0, s], feedback[0, s] = i, f
    return Observation(n=np.array([n]), target=None if target is None else np.array([target]), queries=queries,
                       feedback=feedback, steps=np.array([len(pairs)]), done=np.array([False]))


def _exact_argmin(q: tuple[Fraction, ...]) -> set[int]:
    low = min(q)
    return {i for i, x in enumerate(q) if x == low}


def _all_rows(n: int, v: int) -> EpisodeBatch:
    rows = [(s, j) for s in itertools.combinations(range(v), n) for j in range(n)]  #the uniform law, exactly
    values = np.array([s for s, _ in rows], dtype=np.int64)
    target_idx = np.array([j for _, j in rows], dtype=np.int64)
    return EpisodeBatch(n=np.full(len(rows), n), values=values, target_idx=target_idx,
                        target=values[np.arange(len(rows)), target_idx], adversarial=np.zeros(len(rows), dtype=bool))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_cmp_matches_enumeration(n: int) -> None:
    table = cmp_table(5)
    enum = enumerate_cmp(n)
    for history, (value, q_exact) in enum.items():
        obs = _obs(n, None, history)
        lo, m = (int(x[0]) for x in cmp_state(obs))
        assert Fraction(int(table.total[m]), m) == value
        totals = cmp_q_totals(table, m)
        for i, want in enumerate(q_exact):
            if lo <= i < lo + m:
                assert Fraction(int(totals[i - lo]), m) == want
            else:
                assert want == 1 + value  #repeat or out-of-bracket
        q = q_values(obs, table, 256, n_index=n)[0]
        assert np.allclose(q, [float(x) for x in q_exact], rtol=0, atol=1e-12)
        assert set(np.nonzero(optimal_mask(obs, table, 256, n_index=n)[0])[0]) == _exact_argmin(q_exact)
    assert len(enum) > 0


@pytest.mark.parametrize("n, v", VAL_SIZES)
def test_val_matches_enumeration(tiny: ValTable, n: int, v: int) -> None:
    policy = dp_policy(tiny, v, n_index=n)
    checked, outside = 0, 0
    for target in range(v):
        for history, (value, q_exact) in enumerate_val_uniform(n, v, target).items():
            obs = _obs(n, target, history)
            lo, m, size, t = (int(x[0]) for x in val_state(obs, v))
            assert abs(tiny.value[m, size, t] - float(value)) < 1e-12
            q = q_values(obs, tiny, v, n_index=n)[0]
            assert np.allclose(q, [float(x) for x in q_exact], rtol=0, atol=1e-12)
            for i in range(n):
                if not lo <= i < lo + m:
                    assert q_exact[i] == 1 + value  #brute force over arrays agrees: no information outside the bracket
                    outside += 1
            best = _exact_argmin(q_exact)
            assert set(np.nonzero(optimal_mask(obs, tiny, v, n_index=n)[0])[0]) == best  #float ties == exact ties
            assert policy(obs)[0] == min(best)
            vlo, vhi = (int(x[0]) for x in value_feasible_bracket(obs, v))
            assert all(vlo <= i <= vhi for i in best), (target, sorted(history), sorted(best), (vlo, vhi))
            checked += 1
    assert checked > 0 and (n == 1 or outside > 0)


def test_optimal_positions_value_feasible_whole_table() -> None:
    table = val_uniform_table(10, 30)
    for m in range(1, 11):
        for size in range(m, 31):
            for t in range(1, size + 1):
                q = val_q(table, m, size, t)
                best = np.nonzero(q <= q.min() + TIE_TOL)[0]
                feasible = hypergeom_rank(m, size, t) > 0  #positions where T can be
                assert np.all(feasible[best]), (m, size, t, best)
                assert abs(q.min() - table.value[m, size, t]) < 1e-12 and table.best[m, size, t] == best[0]


@pytest.mark.parametrize("feedback", ["cmp", "val"])
def test_q_outside_bracket_is_one_plus_value_random_histories(mid: ValTable, feedback: str) -> None:
    cfg = EnvConfig(v=64, n_max=16, feedback=feedback, **OPEN)
    table: CmpTable | ValTable = cmp_table(16) if feedback == "cmp" else mid
    policy = dp_policy(table, 64)
    rng = np.random.default_rng(0)
    checked = 0
    for steps in range(8):
        env = SearchEnv(cfg, sample_batch(cfg, 300, rng))
        for _ in range(steps):
            env.step(rng.integers(0, env.observe().n))  #repeats and out-of-bracket queries included
        obs = env.observe()
        rows = np.nonzero(~obs.done)[0]
        q = q_values(obs, table, 64)
        mask = optimal_mask(obs, table, 64)
        lo, hi = bracket(obs)
        if feedback == "cmp":
            value = cmp_value(table, hi - lo + 1)
        else:
            _, m, size, t = val_state(obs, 64)
            value = table.value[m, size, t]
            vlo, vhi = value_feasible_bracket(obs, 64)
        assert np.array_equal(policy(obs)[rows], np.argmax(mask, axis=1)[rows])
        for i in rows:
            idx = np.arange(obs.n[i])
            out = (idx < lo[i]) | (idx > hi[i])
            assert np.all(q[i, idx[out]] == 1.0 + value[i])
            assert abs(q[i, :obs.n[i]].min() - value[i]) < 1e-12 and np.all(np.isinf(q[i, obs.n[i]:]))
            best = np.nonzero(mask[i])[0]
            assert np.all((lo[i] <= best) & (best <= hi[i]))
            if feedback == "val":
                assert np.all((vlo[i] <= best) & (best <= vhi[i]))
            checked += 1
        assert np.all(np.isnan(q[obs.done])) and not mask[obs.done].any()
    assert checked > 1000


def test_outcome_probabilities_sum_to_one() -> None:
    for m in range(1, 7):
        for size in range(m, 21):
            for t in range(1, size + 1):
                prior = hypergeom_rank(m, size, t)
                assert abs(prior.sum() - 1.0) < 1e-12
                for j in range(1, m + 1):
                    cond = val_outcome_given_rank(m, size, t, j)
                    assert np.all(cond[j - 1] == 0.0)
                    live = (np.arange(1, m + 1) != j) & (prior > 0)
                    assert np.allclose(cond[live].sum(axis=1), 1.0, rtol=0, atol=1e-12)
                    assert abs(prior[j - 1] + (prior[:, None] * cond).sum() - 1.0) < 1e-12


def test_cmp_hand() -> None:
    table = cmp_table(64)
    totals = cmp_q_totals(table, 5)
    assert cmp_value(table, 5) == 2.2 and set(np.nonzero(totals == totals.min())[0] + 1) == {2, 3, 4}
    assert table.total[64] == 328 and cmp_value(table, 64) == 5.125  #binary search: 328 / 64


def test_dense_bracket_costs_one() -> None:
    table = val_uniform_table(8, 20)
    for m in range(1, 9):
        assert np.all(table.value[m, m, 1:m + 1] == 1.0)
        assert np.array_equal(table.best[m, m, 1:m + 1], np.arange(m))  #every integer is present: T's index is known


@pytest.mark.parametrize("n, v", [(3, 8), (4, 10), (5, 12)])
def test_dp_policy_attains_value_and_beats_interpolation(tiny: ValTable, n: int, v: int) -> None:
    cfg = EnvConfig(v=v, n_min=n, n_max=n, feedback="val", cap_mode="linear")
    batch = _all_rows(n, v)
    expected = tiny.value[n, v, batch.target + 1].mean()  #initial state (N, V, T + 1) from the sentinels
    used = {}
    for name, policy in (("dp", dp_policy(tiny, v, n_index=n)), ("interpolation", make_policy("interpolation", cfg))):
        env = SearchEnv(cfg, batch)
        rollout(env, policy)
        assert env.success.all()
        used[name] = env.queries_used.mean()
    assert abs(used["dp"] - expected) < 1e-9
    assert used["dp"] <= used["interpolation"] + 1e-12
    cmp_cfg = EnvConfig(v=v, n_min=n, n_max=n, cap_mode="linear")
    env = SearchEnv(cmp_cfg, batch)
    rollout(env, dp_policy(cmp_table(n), v, n_index=n))
    assert env.success.all() and abs(env.queries_used.mean() - cmp_value(cmp_table(n), n)) < 1e-12


def test_save_load_roundtrip(tiny: ValTable, tmp_path) -> None:
    path = tmp_path / "val.npz"
    save_val_table(tiny, path)
    loaded = load_val_table(path)
    assert loaded.n_max == tiny.n_max and loaded.l_max == tiny.l_max
    assert np.array_equal(loaded.value, tiny.value, equal_nan=True) and np.array_equal(loaded.best, tiny.best)
    for name, fields in (("wrong_n.npz", dict(value=tiny.value, best=tiny.best, n_max=6, l_max=tiny.l_max)),
                         ("short_best.npz", dict(value=tiny.value, best=tiny.best[:, :, :-1], n_max=tiny.n_max,
                                                 l_max=tiny.l_max))):
        np.savez(tmp_path / name, **fields)
        with pytest.raises(ValueError):
            load_val_table(tmp_path / name)
    load_val_table(path)


def test_input_validation(tiny: ValTable) -> None:
    for bad in ((0, 4), (5, 4)):
        with pytest.raises(ValueError):
            val_uniform_table(*bad)
    with pytest.raises(ValueError):
        cmp_table(0)
    with pytest.raises(ValueError):
        val_q(tiny, 6, 12, 3)
    with pytest.raises(ValueError):
        enumerate_val_uniform(5, 4, 0)
    cfg = EnvConfig(v=64, n_max=16, feedback="val", **OPEN)
    obs = SearchEnv(cfg, sample_batch(cfg, 50, np.random.default_rng(1))).observe()
    with pytest.raises(ValueError):
        q_values(obs, tiny, 64)  #initial L = 64 exceeds l_max = 12
    with pytest.raises(ValueError):
        dp_policy(tiny, 64)(obs)
    cmp_obs = SearchEnv(EnvConfig(**OPEN), sample_batch(EnvConfig(**OPEN), 5, np.random.default_rng(2))).observe()
    with pytest.raises(ValueError):
        val_state(cmp_obs, 256)  #val table needs the target
