import dataclasses

import numpy as np
import pytest

from config import EnvConfig, GateConfig
from cvar import cvar
from dp import cmp_table, dp_policy, val_uniform_table
from env import EpisodeBatch, Observation, sample_batch
from metrics import (competence_gate, dp_agreement, episode_summary, execution_errors, record_rollout,
                     reference_agreement, step_metrics, summarize_steps, vf_midpoint_costs)
from teachers import Policy, binary_search, interpolation_search, make_policy

VAL, CMP = EnvConfig(feedback="val"), EnvConfig()
NAN = float("nan")


def _batch(rows: list[list[int]], target_idx: list[int], v: int = 256) -> EpisodeBatch:
    n = np.array([len(r) for r in rows])
    values = np.full((len(rows), int(n.max())), v, dtype=np.int64)
    for i, r in enumerate(rows):
        values[i, :len(r)] = r
    t = np.asarray(target_idx, dtype=np.int64)
    return EpisodeBatch(n=n, values=values, target_idx=t, target=values[np.arange(len(rows)), t],
                        adversarial=np.zeros(len(rows), dtype=bool))


def _scripted(scripts: dict[int, list[int]], fallback: Policy = binary_search) -> Policy:
    def policy(obs: Observation) -> np.ndarray:
        action = np.asarray(fallback(obs)).copy()
        for row, queries in scripts.items():
            if obs.steps[row] < len(queries):
                action[row] = queries[obs.steps[row]]
        return action
    return policy


def _eq(got: np.ndarray, want: list) -> bool:
    return np.allclose(np.asarray(got, dtype=float), np.asarray(want, dtype=float), rtol=0, atol=1e-12, equal_nan=True)


def test_hand_episode_val() -> None:
    batch = _batch([[10, 20, 30, 40, 50, 60, 70, 80], [10, 11, 12, 15, 16]], [5, 2])
    rec = record_rollout(VAL, batch, _scripted({0: [3, 3, 0, 7, 5], 1: [0, 4, 3, 1, 2]}))
    assert rec.success.all() and rec.final.steps.tolist() == [5, 5] and rec.cost.tolist() == [5, 5]
    sm = step_metrics(rec)
    k = 5
    assert sm.taken[:, :k].all() and not sm.taken[:, k:].any()
    #row 0, T = 60: q=3 (40 < T, bracket 8 -> 4), repeat of 3, q=0 outside, q=7 (80 > T, 4 -> 3), q=5 hit
    assert sm.in_bracket[0, :k].tolist() == [True, False, False, True, True]
    assert sm.repeated[0, :k].tolist() == [False, True, False, False, False]
    assert sm.in_value_feasible[0, :k].tolist() == [True, False, False, True, True]
    assert _eq(sm.bracket_eliminated[0, :k], [0.5, 0.0, 0.0, 0.25, NAN])
    assert _eq(sm.value_feasible_eliminated[0, :k], [0.5, 0.0, 0.0, 0.25, NAN])
    assert sm.hit[0, :k].tolist() == [False] * 4 + [True]
    assert sm.bracket_size[0, :k].tolist() == [8, 4, 4, 4, 3] and sm.remaining[0, :k].tolist() == [8, 7, 6, 5, 4]
    #row 1, T = 12: after A[0] = 10 the value-feasible range is already (1, 2) (only 11 lies between 10 and 12), so
    #q=4 and q=3 are inside the comparison bracket but outside the value-feasible range
    assert sm.in_bracket[1, :k].tolist() == [True, True, True, True, True]
    assert sm.in_value_feasible[1, :k].tolist() == [True, False, False, True, True]
    assert _eq(sm.bracket_eliminated[1, :k], [0.2, 0.25, 1 / 3, 0.5, NAN])
    assert _eq(sm.value_feasible_eliminated[1, :k], [0.6, 0.0, 0.0, 0.5, NAN])
    summary = summarize_steps(sm)
    assert summary["queries"] == 10 and summary["in_bracket_rate"] == 0.8 and summary["repeat_rate"] == 0.1
    assert abs(summary["out_of_value_feasible_rate"] - 0.4) < 1e-12
    assert _eq(summary["hit_rate_by_step"][:k], [0, 0, 0, 0, 1])
    assert execution_errors(rec).tolist() == [True, True]  #row 0 repeats and leaves the range; row 1 leaves the range


def test_certification_cases() -> None:
    batch = _batch([[10, 20, 30, 40], [10, 20, 30, 40], [10, 20]], [3, 3, 1])
    rec = record_rollout(CMP, batch, _scripted({0: [1, 3], 1: [0, 0, 3], 2: [0, 0]}))
    assert rec.success.tolist() == [True, True, False] and rec.cost.tolist() == [2, 3, 4]  #row 2: 2 + K(2)
    table = cmp_table(4)
    sm = step_metrics(rec, table)
    assert sm.certified[0, :2].tolist() == [True, True]  #distinct in-bracket with m = r = 4, then m = 2 <= r = 3
    assert sm.bracket_size[1, 1] == 3 and sm.remaining[1, 1] == 3 and not sm.certified[1, 1]  #repeat with m = r
    assert not sm.certified[1, 2]  #after the wasted query m = 3 > r = 2: V(s) itself is not exact
    assert sm.remaining[2, 1] == 1 and sm.bracket_size[2, 1] == 1 and not sm.certified[2, 1]  #repeat on the last query
    assert sm.certified[1, 0] and sm.certified[2, 0]
    assert abs(sm.regret[1, 1] - 1.0) < 1e-12 and abs(sm.regret[2, 1] - 1.0) < 1e-12  #Q = 1 + V(s) for a repeat
    summary = summarize_steps(sm)
    assert summary["certified_steps"] == 4 and summary["uncertified_steps"] == 3  #2 + 1 + 1 of 2 + 3 + 2 steps


def test_regret_against_dp() -> None:
    cfg = EnvConfig(v=64, n_max=16, feedback="val")
    table = val_uniform_table(16, 64)
    batch = sample_batch(cfg, 400, np.random.default_rng(0))
    dp = step_metrics(record_rollout(cfg, batch, dp_policy(table, 64)), table)
    assert np.all(np.abs(dp.regret[dp.taken]) < 1e-12) and dp.certified[dp.taken].all()
    binary = step_metrics(record_rollout(cfg, batch, make_policy("binary", cfg)), table)
    assert np.all(binary.regret[binary.taken] > -1e-12) and binary.regret[binary.taken].mean() > 0.05
    rng = np.random.default_rng(1)
    rand = step_metrics(record_rollout(cfg, batch, lambda obs: rng.integers(0, obs.n)), table)
    outside = rand.taken & ~rand.in_bracket
    assert outside.sum() > 100 and np.allclose(rand.regret[outside], 1.0, rtol=0, atol=1e-12)  #Q = 1 + V(s)
    assert (rand.taken & ~rand.certified).any() and np.isnan(rand.regret[~rand.taken]).all()
    cmp_cfg = dataclasses.replace(cfg, feedback="cmp")
    cmp = step_metrics(record_rollout(cmp_cfg, batch, binary_search), cmp_table(16))
    assert np.all(np.abs(cmp.regret[cmp.taken]) < 1e-12)  #binary search is cmp-optimal
    assert cmp.in_value_feasible is None and cmp.value_feasible_eliminated is None


def test_episode_summary() -> None:
    cfg = EnvConfig(v=256, n_min=16, n_max=16, feedback="val")
    rows = [list(range(0, 160, 10))] * 4
    batch = _batch(rows, [10, 7, 0, 7])
    rec = record_rollout(cfg, batch, _scripted({0: list(range(11)), 2: [1] * 16}))  #11-query scan; row 2 never hits
    s = episode_summary(rec, (0.25, 0.5, 1.0))
    costs = rec.cost.tolist()
    assert costs[0] == 11 and costs[2] == 32 and rec.success.tolist() == [True, True, False, True]
    assert s.episodes == 4 and s.success_rate == 0.75 and s.failure_rate == 0.25 and s.failure_cost == 32.0
    assert s.mean_cost == np.mean(costs) and s.mean_queries_success == np.mean([costs[0], costs[1], costs[3]])
    assert s.cvar[0.25] == 32.0 and s.cvar[0.5] == (32.0 + 11.0) / 2 and s.cvar[1.0] == np.mean(costs)
    assert s.cost_histogram[32] == 1 and s.cost_histogram[11] == 1 and s.cost_histogram.sum() == 4
    assert s.over_log_cap == 0.5  #the 11-query success (log cap K(16) = 10) and the failure
    small = record_rollout(VAL, _batch([[10, 20, 30, 40]] * 2, [0, 0]), _scripted({0: [1] * 4}))
    assert small.success.tolist() == [False, True] and small.final.steps[0] == 4  #4 steps, within the log cap of 6
    assert episode_summary(small, (1.0,)).over_log_cap == 0.5  #a failure counts even when it used fewer queries


def _gate_records() -> tuple:
    rows = [list(range(10, 90, 10))] * 200  #N = 8: K(8) = 8 leaves room for the five-query error episode
    batch = _batch(rows, [i % 8 for i in range(200)])
    error = _scripted({1: [0, 0, 0, 0, 1]})  #row 1 (T = 20): three repeats, cost 5, the unique maximum
    greedy, clean = record_rollout(VAL, batch, error), record_rollout(VAL, batch, binary_search)
    return batch, greedy, clean, vf_midpoint_costs(VAL, batch)


def test_competence_gate_known_share() -> None:
    batch, greedy, clean, ref = _gate_records()
    assert ref[:8].tolist() == [3, 2, 3, 1, 3, 2, 3, 4] and greedy.cost[1] == 5 and clean.cost.max() == 4
    alpha = 1 / 200  #k = 1: CVaR is the maximum, so the execution share is 5 - 4 = 1 exactly
    assert execution_errors(greedy).tolist() == [False, True] + [False] * 198
    result = competence_gate(greedy, clean, ref, alpha, 10.0, GateConfig())
    value, threshold, ok = result.criteria["execution_cvar_share_greedy"]
    assert value == 1.0 and threshold == 2.5 and ok
    assert result.criteria["execution_cvar_share_sampled"][0] == 0.0
    assert result.criteria["repeat_rate_greedy"][0] == 3 / greedy.final.steps.sum() < 0.01
    assert result.criteria["sampled_greedy_gap"][0] == -1.0 and result.passed
    tight = competence_gate(greedy, clean, ref, alpha, 2.0, GateConfig())  #threshold 0.25 * 2 = 0.5 < 1
    assert not tight.passed and not tight.criteria["execution_cvar_share_greedy"][2]
    assert all(ok for name, (_, _, ok) in tight.criteria.items() if name != "execution_cvar_share_greedy")
    flipped = competence_gate(clean, greedy, ref, alpha, 2.0, GateConfig())  #sampled worse than greedy by 1 > 0.5
    assert not flipped.criteria["sampled_greedy_gap"][2]


def test_competence_gate_fail_cases() -> None:
    batch, greedy, clean, ref = _gate_records()
    alpha = 0.1
    fails = record_rollout(VAL, batch, _scripted({0: [1] * 8}))  #row 0 (T = 10) repeats 1 until the cap
    result = competence_gate(clean, fails, ref, alpha, 10.0, GateConfig())
    assert not result.criteria["failure_rate_sampled"][2] and result.criteria["failure_rate_greedy"][2]
    sloppy = record_rollout(VAL, batch, _scripted({i: [3, 3, 0, 1] for i in range(0, 80, 8)}))  #10 rows, T = 10
    result = competence_gate(clean, sloppy, ref, alpha, 10.0, GateConfig())
    assert not result.criteria["repeat_rate_sampled"][2] and not result.criteria["out_of_range_rate_sampled"][2]
    with pytest.raises(ValueError):
        competence_gate(clean, clean, ref, alpha, 0.0, GateConfig())
    with pytest.raises(ValueError):
        competence_gate(clean, clean, ref[:10], alpha, 1.0, GateConfig())
    other = record_rollout(VAL, _batch([[11] + list(range(20, 90, 10))] * 200, [i % 8 for i in range(200)]),
                           binary_search)
    with pytest.raises(ValueError):
        competence_gate(clean, other, ref, alpha, 1.0, GateConfig())
    cmp_rec = record_rollout(CMP, batch, binary_search)
    with pytest.raises(ValueError):
        competence_gate(cmp_rec, cmp_rec, ref, alpha, 1.0, GateConfig())


def test_agreement() -> None:
    cfg = EnvConfig(v=64, n_max=16, feedback="val", distribution="mixture", adversarial="trap", p=0.3)
    batch = sample_batch(cfg, 400, np.random.default_rng(2))
    interp = make_policy("interpolation", cfg)
    rec = record_rollout(cfg, batch, interp)
    taken = step_metrics(rec).taken
    agree = reference_agreement(rec, {"interpolation": interp, "binary": binary_search,
                                      "direct": lambda obs: interpolation_search(obs, 64)})
    assert agree["interpolation"][taken].all() and agree["direct"][taken].all()
    assert not agree["interpolation"][~taken].any()
    assert 0.0 < agree["binary"][taken].mean() < 1.0
    table = val_uniform_table(16, 64)
    dp_rec = record_rollout(cfg, batch, dp_policy(table, 64))
    dp_taken = step_metrics(dp_rec).taken
    assert dp_agreement(dp_rec, table)[dp_taken].all()
    assert 0.0 < dp_agreement(rec, table)[taken].mean() < 1.0


def test_summary_cvar_matches_cvar_module() -> None:
    cfg = EnvConfig(v=64, n_max=16, feedback="val", distribution="adversarial", adversarial="trap")
    rec = record_rollout(cfg, sample_batch(cfg, 300, np.random.default_rng(3)), make_policy("interpolation", cfg))
    s = episode_summary(rec, (0.05, 0.1, 0.25))
    assert all(s.cvar[a] == cvar(rec.cost, a) for a in (0.05, 0.1, 0.25))
