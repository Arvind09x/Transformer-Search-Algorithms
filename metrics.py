import dataclasses
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from config import EnvConfig, GateConfig
from cvar import cvar
from dp import CmpTable, ValTable, optimal_mask, q_values
from env import EQUAL, EpisodeBatch, Observation, SearchEnv, rollout
from teachers import Policy, bracket, observation_prefix, value_feasible_bracket, value_feasible_midpoint


@dataclass(frozen=True)
class EpisodeRecord:
    cfg: EnvConfig
    batch: EpisodeBatch  #hidden arrays: used for reference costs and counterfactuals, never by a policy
    final: Observation  #full history at the end of every episode
    success: np.ndarray  #(B,)
    cost: np.ndarray  #(B,) C = Q + failure_cost_coef * K(N) * 1[failure]
    cap: np.ndarray  #(B,) K(N)


@dataclass(frozen=True)
class EpisodeSummary:
    episodes: int
    success_rate: float
    failure_rate: float
    failure_cost: float  #mean cost of failed episodes; NaN without failures
    mean_cost: float
    mean_queries_success: float
    cvar: dict[float, float]
    cost_histogram: np.ndarray  #count of episodes at each integer cost
    over_log_cap: float  #fraction that would fail under the logarithmic cap (plan Section 11.5)


@dataclass(frozen=True)
class StepMetrics:  #(B, k_max) arrays over decision steps; meaningful where taken
    taken: np.ndarray
    hit: np.ndarray
    in_bracket: np.ndarray
    repeated: np.ndarray
    in_value_feasible: np.ndarray | None  #None under cmp
    bracket_eliminated: np.ndarray  #fraction of the comparison bracket removed; NaN on hits and untaken steps
    value_feasible_eliminated: np.ndarray | None
    bracket_size: np.ndarray  #m before the query
    remaining: np.ndarray  #r = K(N) - queries already made
    regret: np.ndarray  #Q(s, a) - V(s) against the uncapped DP table; NaN without a table
    certified: np.ndarray  #regret exact under the cap (plan Section 9)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    criteria: dict[str, tuple[float, float, bool]]  #name -> (value, threshold, value < threshold)


def record_rollout(cfg: EnvConfig, batch: EpisodeBatch, policy: Policy) -> EpisodeRecord:
    env = SearchEnv(cfg, batch)
    rollout(env, policy)
    return EpisodeRecord(cfg=cfg, batch=batch, final=env.observe(), success=env.success, cost=env.cost(), cap=env.cap)


def vf_midpoint_costs(cfg: EnvConfig, batch: EpisodeBatch) -> np.ndarray:
    return record_rollout(cfg, batch, lambda obs: value_feasible_midpoint(obs, cfg.v)).cost  #gate (iii) reference


def _log_cap(cfg: EnvConfig, n: np.ndarray) -> np.ndarray:
    log_cfg = dataclasses.replace(cfg, cap_mode="log")
    return np.array([log_cfg.query_cap(int(m)) for m in n])


def episode_summary(record: EpisodeRecord, alphas: tuple[float, ...]) -> EpisodeSummary:
    failed = ~record.success
    steps = record.final.steps
    over = failed | (steps > _log_cap(record.cfg, record.final.n))  #not found within the log-cap budget
    return EpisodeSummary(
        episodes=len(record.cost), success_rate=float(record.success.mean()), failure_rate=float(failed.mean()),
        failure_cost=float(record.cost[failed].mean()) if failed.any() else float("nan"),
        mean_cost=float(record.cost.mean()),
        mean_queries_success=float(steps[record.success].mean()) if record.success.any() else float("nan"),
        cvar={a: cvar(record.cost, a) for a in alphas}, cost_histogram=np.bincount(record.cost.astype(np.int64)),
        over_log_cap=float(over.mean()))


def _states(record: EpisodeRecord) -> Iterator[tuple[int, np.ndarray, Observation, Observation]]:
    obs = record.final
    for t in range(obs.queries.shape[1]):
        rows = obs.steps > t  #rows that took action t + 1 from the state after t queries
        if not rows.any():
            return
        before = observation_prefix(obs, np.minimum(t, obs.steps))
        after = observation_prefix(obs, np.minimum(t + 1, obs.steps))
        yield t, rows, dataclasses.replace(before, done=~rows), after  #the prefix keeps the final done flag; reset it


def step_metrics(record: EpisodeRecord, table: CmpTable | ValTable | None = None) -> StepMetrics:
    obs, v = record.final, record.cfg.v
    size, k_max = obs.queries.shape
    val = obs.target is not None
    taken = np.arange(k_max)[None, :] < obs.steps[:, None]
    nan = np.full((size, k_max), np.nan)
    out = dict(hit=np.zeros_like(taken), in_bracket=np.zeros_like(taken), repeated=np.zeros_like(taken),
               in_value_feasible=np.zeros_like(taken) if val else None, bracket_eliminated=nan.copy(),
               value_feasible_eliminated=nan.copy() if val else None, bracket_size=np.zeros((size, k_max), np.int64),
               remaining=np.zeros((size, k_max), np.int64), regret=nan.copy(), certified=np.zeros_like(taken))
    for t, rows, before, after in _states(record):
        q = obs.queries[:, t]
        hit = obs.feedback[:, t] == (obs.target if val else EQUAL)
        lo, hi = bracket(before)
        lo2, hi2 = bracket(after)
        m, m2 = hi - lo + 1, hi2 - lo2 + 1
        inside = (lo <= q) & (q <= hi)
        miss = rows & ~hit
        out["hit"][:, t] = rows & hit
        out["in_bracket"][:, t] = rows & inside
        out["repeated"][:, t] = rows & (obs.queries[:, :t] == q[:, None]).any(axis=1)
        out["bracket_eliminated"][miss, t] = ((m - m2) / m)[miss]
        out["bracket_size"][:, t] = np.where(rows, m, 0)
        remaining = record.cap - t
        out["remaining"][:, t] = np.where(rows, remaining, 0)
        out["certified"][:, t] = rows & (m <= remaining) & (inside | (m <= remaining - 1))  #in-bracket implies distinct
        if val:
            vlo, vhi = value_feasible_bracket(before, v)
            vlo2, vhi2 = value_feasible_bracket(after, v)
            w, w2 = vhi - vlo + 1, vhi2 - vlo2 + 1
            out["in_value_feasible"][:, t] = rows & (vlo <= q) & (q <= vhi)
            out["value_feasible_eliminated"][miss, t] = ((w - w2) / w)[miss]
        if table is not None:
            qv = q_values(before, table, v)
            idx = np.nonzero(rows)[0]
            out["regret"][idx, t] = qv[idx, q[idx]] - qv[idx].min(axis=1)  #V(s) = min over legal actions
    return StepMetrics(taken=taken, **out)


def summarize_steps(sm: StepMetrics) -> dict[str, object]:
    taken = sm.taken
    queries = int(taken.sum())

    def by_step(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        counts = mask.sum(axis=0)
        return np.where(counts > 0, np.where(mask, x, 0.0).sum(axis=0) / np.maximum(counts, 1), np.nan)

    regret_known = taken & ~np.isnan(sm.regret)
    certified, uncertified = regret_known & sm.certified, regret_known & ~sm.certified
    out: dict[str, object] = dict(
        queries=queries, in_bracket_rate=float(sm.in_bracket[taken].mean()),
        repeat_rate=float(sm.repeated[taken].mean()),
        bracket_eliminated_by_step=by_step(np.nan_to_num(sm.bracket_eliminated), ~np.isnan(sm.bracket_eliminated)),
        hit_rate_by_step=by_step(sm.hit.astype(float), taken),
        regret_certified_mean=float(sm.regret[certified].mean()) if certified.any() else float("nan"),
        regret_uncertified_mean=float(sm.regret[uncertified].mean()) if uncertified.any() else float("nan"),
        certified_steps=int(certified.sum()), uncertified_steps=int(uncertified.sum()))
    if sm.in_value_feasible is not None:
        out["out_of_value_feasible_rate"] = float(1.0 - sm.in_value_feasible[taken].mean())
        vfe = sm.value_feasible_eliminated
        out["value_feasible_eliminated_by_step"] = by_step(np.nan_to_num(vfe), ~np.isnan(vfe))
    return out


def reference_agreement(record: EpisodeRecord, policies: dict[str, Policy]) -> dict[str, np.ndarray]:
    obs = record.final
    agree = {name: np.zeros(obs.queries.shape, dtype=bool) for name in policies}
    for t, rows, before, _ in _states(record):
        q = obs.queries[:, t]
        for name, policy in policies.items():
            agree[name][:, t] = rows & (policy(before) == q)
    return agree  #(B, k_max) per reference; the reference acts on the history preceding q_t


def dp_agreement(record: EpisodeRecord, table: CmpTable | ValTable) -> np.ndarray:
    obs = record.final
    agree = np.zeros(obs.queries.shape, dtype=bool)
    for t, rows, before, _ in _states(record):
        mask = optimal_mask(before, table, record.cfg.v)
        agree[:, t] = rows & mask[np.arange(len(rows)), obs.queries[:, t]]  #any DP-optimal action counts: ties
    return agree


def execution_errors(record: EpisodeRecord) -> np.ndarray:
    sm = step_metrics(record)
    errors = ~record.success | (sm.taken & sm.repeated).any(axis=1)
    if sm.in_value_feasible is not None:
        errors |= (sm.taken & ~sm.in_value_feasible).any(axis=1)
    return errors  #(B,) failed, repeated a query, or queried outside the value-feasible range


def competence_gate(greedy: EpisodeRecord, sampled: EpisodeRecord, reference_cost: np.ndarray, alpha: float,
                    g_ref: float, gate: GateConfig) -> GateResult:
    if greedy.final.target is None:
        raise ValueError("the competence gate is defined under val feedback")
    if g_ref <= 0.0:
        raise ValueError(f"g_ref={g_ref} must be positive: the gate is read against a real reference gap")
    for rec in (greedy, sampled):
        if not np.array_equal(rec.batch.values, greedy.batch.values) or len(reference_cost) != len(rec.cost):
            raise ValueError("greedy, sampled and reference costs must be on the same arrays")
    criteria: dict[str, tuple[float, float, bool]] = {}

    def check(name: str, value: float, threshold: float) -> None:
        criteria[name] = (value, threshold, value < threshold)

    for label, rec in (("greedy", greedy), ("sampled", sampled)):
        summary = summarize_steps(step_metrics(rec))
        check(f"failure_rate_{label}", float((~rec.success).mean()), gate.failure_rate_frac_of_alpha * alpha)
        check(f"repeat_rate_{label}", float(summary["repeat_rate"]), gate.repeat_rate_max)
        check(f"out_of_range_rate_{label}", float(summary["out_of_value_feasible_rate"]), gate.out_of_range_rate_max)
        cleaned = np.where(execution_errors(rec), reference_cost, rec.cost)  #errors take the vf-midpoint cost
        check(f"execution_cvar_share_{label}", cvar(rec.cost, alpha) - cvar(cleaned, alpha),
              gate.execution_cvar_share_max * g_ref)
    check("sampled_greedy_gap", cvar(sampled.cost, alpha) - cvar(greedy.cost, alpha),
          gate.sampled_greedy_gap_max * g_ref)
    return GateResult(passed=all(ok for _, _, ok in criteria.values()), criteria=criteria)
