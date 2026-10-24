import dataclasses
import json
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

PARAM_COUNT: int | None = None  #exact count from model.py, recorded once it exists

FeedbackMode = Literal["cmp", "val"]
Distribution = Literal["uniform", "adversarial", "mixture"]
Adversarial = Literal["clustered", "trap"]
Objective = Literal["mean", "cvar"]
EvalMode = Literal["greedy", "sampled"]
Phase = Literal["0", "A", "B", "C1", "C2", "D", "E"]
CapMode = Literal["log", "linear"]


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _literal_values(tp: Any) -> tuple[Any, ...] | None:
    if get_origin(tp) is Literal:
        return get_args(tp)
    if get_origin(tp) in (Union, types.UnionType):
        values: list[Any] = []
        for arg in get_args(tp):
            if arg is type(None):
                values.append(None)
            elif get_origin(arg) is Literal:
                values.extend(get_args(arg))
            else:
                return None
        return tuple(values)
    return None


def _check_literals(obj: Any) -> None:
    for f in dataclasses.fields(obj):
        allowed = _literal_values(f.type)
        value = getattr(obj, f.name)
        _require(allowed is None or value in allowed,
                 f"{type(obj).__name__}.{f.name}={value!r} not in {allowed}")


def _positive_ints(obj: Any, names: tuple[str, ...]) -> None:
    for name in names:
        value = getattr(obj, name)
        _require(isinstance(value, int) and value > 0, f"{type(obj).__name__}.{name}={value!r} must be a positive int")


@dataclass(frozen=True)
class EnvConfig:
    v: int = 256  #values lie in [0, v)
    n_min: int = 4
    n_max: int = 64
    feedback: FeedbackMode = "cmp"
    distribution: Distribution = "uniform"
    adversarial: Adversarial | None = None  #required unless uniform; Phase 0 selects
    p: float = 0.10  #used only under mixture
    clustered_k: int = 2  #number of cluster centres
    clustered_sigma: float = 8.0  #at N = 64, sigma = 8 makes the two clusters near-contiguous runs; Phase 0 sweeps {8, 16, 32}
    trap_strength: float = 0.75  #s; 0 gives the widest window; the uniform law is distribution = uniform; 1 a point mass at the extreme rank (pre-emptable by one probe)
    cap_mode: CapMode = "linear"  #linear: K(N) = N, a failure is an execution error; log: deadline variant (plan 11.5)
    cap_mult: int = 2  #log only: K(N) = cap_mult * ceil(log2 N) + cap_add
    cap_add: int = 2
    query_cost: int = 1  #C = query_cost * Q + failure_cost_coef * K(N) * 1[failure]
    failure_cost_coef: int = 1
    discount: float = 1.0

    def __post_init__(self) -> None:
        _check_literals(self)
        _positive_ints(self, ("v", "n_min", "n_max"))
        _require(self.n_min <= self.n_max <= self.v, "need n_min <= n_max <= v")
        _require(self.distribution == "uniform" or self.adversarial is not None,
                 f"distribution={self.distribution!r} needs adversarial set")
        _require(0.0 <= self.p <= 1.0, "p must lie in [0, 1]")
        _positive_ints(self, ("clustered_k",))
        _require(self.clustered_sigma > 0.0, "clustered_sigma must be positive")
        _require(0.0 <= self.trap_strength <= 1.0, "trap_strength must lie in [0, 1]")
        _require(self.cap_mult >= 0 and self.cap_add >= 1, "cap must be at least 1 query")
        _require(self.query_cost > 0 and self.failure_cost_coef >= 0, "costs must be non-negative, query cost positive")
        _require(0.0 < self.discount <= 1.0, "discount must lie in (0, 1]")

    def query_cap(self, n: int) -> int:
        _require(n >= 1, f"n={n} must be >= 1")
        if self.cap_mode == "linear":
            return n
        return self.cap_mult * (n - 1).bit_length() + self.cap_add  #(n-1).bit_length() == ceil(log2 n), exact


@dataclass(frozen=True)
class ModelConfig:
    n_layers: int = 4
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    activation: Literal["gelu"] = "gelu"
    norm: Literal["pre_ln"] = "pre_ln"
    pos_emb: Literal["learned"] = "learned"
    dropout: float = 0.0
    context_len: int = 130  #2 + 2 * K(64) under K(N) = N; 32 suffices under the log cap
    n_index_logits: int = 64  #fixed head size, independent of the run's n_max; larger N needs a different action representation (plan 11.4)

    def __post_init__(self) -> None:
        _check_literals(self)
        _positive_ints(self, ("n_layers", "d_model", "n_heads", "d_ff", "context_len", "n_index_logits"))
        _require(self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads")
        _require(0.0 <= self.dropout < 1.0, "dropout must lie in [0, 1)")


@dataclass(frozen=True)
class OptimConfig:
    optimizer: Literal["adamw"] = "adamw"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0  #global grad-norm
    lr_schedule: Literal["constant"] = "constant"
    lr_imitation: float = 3e-4
    lr_rl: float = 1e-4
    lr_sweep: tuple[float, ...] = (3e-5, 1e-4, 3e-4)  #pilot only, if RL fails to learn
    grad_aggregation: Literal["sum_steps_mean_episodes"] = "sum_steps_mean_episodes"

    def __post_init__(self) -> None:
        _check_literals(self)
        _require(len(self.betas) == 2 and all(0.0 <= b < 1.0 for b in self.betas), "betas must be two values in [0, 1)")
        _require(self.eps > 0 and self.weight_decay >= 0 and self.grad_clip > 0, "need eps > 0, weight_decay >= 0, grad_clip > 0")
        _require(self.lr_imitation > 0 and self.lr_rl > 0, "learning rates must be positive")
        _require(len(self.lr_sweep) > 0 and all(lr > 0 for lr in self.lr_sweep), "lr_sweep must be non-empty and positive")


@dataclass(frozen=True)
class RLConfig:
    objective: Objective = "mean"
    estimator: Literal["reinforce_rtg"] = "reinforce_rtg"
    baseline: Literal["lagged_ema_by_n_step"] = "lagged_ema_by_n_step"  #indexed by (N, decision step)
    baseline_old_weight: float = 0.95  #new = 0.95 old + 0.05 batch, after each batch
    entropy_coef: float = 0.01
    entropy_hold_frac: float = 0.8  #held for this fraction of updates, then linear to 0
    curriculum: bool = False
    curriculum_stages: tuple[tuple[int, int], ...] = ((4, 8), (4, 16), (4, 32), (4, 64))
    entropy_restart_at_branch: bool = True  #Phase D: entropy schedule restarts at branching over the Phase D budget, identically in both branches

    def __post_init__(self) -> None:
        _check_literals(self)
        _require(0.0 <= self.baseline_old_weight <= 1.0, "baseline_old_weight must lie in [0, 1]")
        _require(self.entropy_coef >= 0.0, "entropy_coef must be non-negative")
        _require(0.0 <= self.entropy_hold_frac <= 1.0, "entropy_hold_frac must lie in [0, 1]")
        _require(len(self.curriculum_stages) > 0, "curriculum_stages must be non-empty")
        for lo, hi in self.curriculum_stages:
            _require(1 <= lo <= hi, f"curriculum stage ({lo}, {hi}) invalid")
        his = [hi for _, hi in self.curriculum_stages]
        _require(his == sorted(his), "curriculum stages must not shrink")


@dataclass(frozen=True)
class CVaRConfig:
    alpha: float = 0.10
    alpha_followups: tuple[float, ...] = (0.05, 0.25)
    mean_weight: float = 0.0  #objective (1 - w) CVaR + w mean; 0 is pure CVaR, anything else is a changed objective and is reported as such

    def __post_init__(self) -> None:
        for a in (self.alpha, *self.alpha_followups):
            _require(0.0 < a <= 1.0, f"alpha={a} must lie in (0, 1]")
        _require(0.0 <= self.mean_weight < 1.0, "mean_weight must lie in [0, 1)")


@dataclass(frozen=True)
class BudgetConfig:
    batch_imitation: int = 256
    batch_pilot: int = 256
    batch_main: int = 1000  #matched across mean and CVaR
    imitation_updates: int = 10_000  #cap
    val_every_imitation: int = 250
    imitation_stop_tol: float = 0.5  #closed-loop mean queries <= teacher mean + tol on the same arrays
    imitation_stop_patience: int = 3  #consecutive validations
    pilot_updates: int = 1000
    pilot_seeds: int = 2
    val_every_pilot: int = 100
    main_updates: int = 5000
    main_seeds: int = 5
    val_every_main: int = 250

    def __post_init__(self) -> None:
        _positive_ints(self, tuple(f.name for f in dataclasses.fields(self) if f.name != "imitation_stop_tol"))
        _require(self.imitation_stop_tol >= 0.0, "imitation_stop_tol must be non-negative")
        for updates, every in ((self.imitation_updates, self.val_every_imitation),
                               (self.pilot_updates, self.val_every_pilot),
                               (self.main_updates, self.val_every_main)):
            _require(updates % every == 0, f"validation interval {every} must divide {updates} updates")


@dataclass(frozen=True)
class EvalConfig:
    val_data_seed: int = 1000  #fixed across all runs
    test_data_seed: int = 2000  #fixed across all runs
    val_episodes: int = 2000
    test_episodes: int = 10_000
    modes: tuple[EvalMode, ...] = ("greedy", "sampled")

    def __post_init__(self) -> None:
        _positive_ints(self, ("val_episodes", "test_episodes"))
        _require(len(self.modes) > 0 and all(m in get_args(EvalMode) for m in self.modes), f"modes={self.modes!r} invalid")


@dataclass(frozen=True)
class Phase0Config:
    ns: tuple[int, ...] = (32, 64)
    alphas: tuple[float, ...] = (0.05, 0.10, 0.25)
    p_grid: tuple[float, ...] = (0.02, 0.05, 0.10, 0.25)
    clustered_sigma_grid: tuple[float, ...] = (8.0, 16.0, 32.0)  #clustered sigma sweep
    episodes: int = 200_000  #per reference policy per condition
    sim_seed: int = 3000
    fresh_seed: int = 3001  #validation of the frozen condition on fresh samples
    guard_frac: float = 0.25  #g of guarded interpolation; clamp to [vlo + floor(g w), vhi - floor(g w)]
    interp_k_grid: tuple[int, ...] = (1, 2)  #k of interpolate-k-then-value-feasible-midpoint
    switch_threshold: float = 0.5  #theta of the adaptive switch: minimum eliminated fraction of the value-feasible range
    bootstrap_resamples: int = 1000  #Monte Carlo uncertainty of the reference gaps

    def __post_init__(self) -> None:
        _positive_ints(self, ("episodes", "bootstrap_resamples"))
        _require(len(self.ns) > 0 and all(isinstance(n, int) and n >= 1 for n in self.ns), "ns must be positive ints")
        _require(all(0.0 < a <= 1.0 for a in self.alphas), "alphas must lie in (0, 1]")
        _require(all(0.0 <= p <= 1.0 for p in self.p_grid), "p_grid must lie in [0, 1]")
        _require(len(self.clustered_sigma_grid) > 0 and all(s > 0.0 for s in self.clustered_sigma_grid),
                 "clustered_sigma_grid must be non-empty and positive")
        _require(0.0 <= self.guard_frac < 0.5, "guard_frac must lie in [0, 0.5)")
        _require(len(self.interp_k_grid) > 0 and all(isinstance(k, int) and k >= 1 for k in self.interp_k_grid),
                 "interp_k_grid must be non-empty ints >= 1")
        _require(0.0 < self.switch_threshold <= 1.0, "switch_threshold must lie in (0, 1]")


@dataclass(frozen=True)
class GateConfig:
    failure_rate_frac_of_alpha: float = 0.01  #failure rate must be below this times cvar.alpha (alpha / 100, plan Section 8)
    repeat_rate_max: float = 0.01  #repeated queries, fraction of all queries
    out_of_range_rate_max: float = 0.01  #queries outside the value-feasible range, fraction of all queries
    execution_cvar_share_max: float = 0.25  #execution-error share of CVaR, fraction of G_ref
    sampled_greedy_gap_max: float = 0.25  #sampled minus greedy CVaR, fraction of G_ref

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            _require(0.0 <= value <= 1.0, f"GateConfig.{f.name}={value!r} must lie in [0, 1]")


@dataclass(frozen=True)
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    cvar: CVaRConfig = field(default_factory=CVaRConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    phase0: Phase0Config = field(default_factory=Phase0Config)
    gate: GateConfig = field(default_factory=GateConfig)
    train_seed: int = 0  #per run
    phase: Phase | None = None  #run label only; changes no setting

    def __post_init__(self) -> None:
        _check_literals(self)
        env, model = self.env, self.model
        _require(env.n_max <= model.n_index_logits, f"n_max={env.n_max} exceeds n_index_logits={model.n_index_logits}")
        _require(2 + 2 * env.query_cap(env.n_max) <= model.context_len,
                 f"context_len={model.context_len} below 2 + 2 * K({env.n_max})")
        for n in self.phase0.ns:
            _require(n <= model.n_index_logits and n <= env.v, f"phase0 n={n} exceeds index head or v")
            _require(2 + 2 * env.query_cap(n) <= model.context_len, f"context_len too short for phase0 n={n}")
        if self.rl.curriculum:
            stages = self.rl.curriculum_stages
            _require(stages[-1] == (env.n_min, env.n_max), f"last curriculum stage {stages[-1]} must equal (n_min, n_max)")
        data_seeds = (self.eval.val_data_seed, self.eval.test_data_seed, self.phase0.sim_seed, self.phase0.fresh_seed)
        _require(len(set(data_seeds)) == len(data_seeds), f"data seeds {data_seeds} must be distinct")
        _require(self.train_seed not in data_seeds, f"train_seed={self.train_seed} collides with a data seed")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return _build(cls, data)

    @classmethod
    def load_json(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _as_tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_as_tuples(v) for v in value)
    return value


def _build(cls: type, data: dict[str, Any]) -> Any:
    fields = {f.name: f for f in dataclasses.fields(cls)}
    _require(set(data) == set(fields),  #strict: a loaded config is exactly what was run
             f"{cls.__name__}: unknown keys {sorted(set(data) - set(fields))}, missing keys {sorted(set(fields) - set(data))}")
    kwargs = {}
    for name, f in fields.items():
        if dataclasses.is_dataclass(f.type):
            kwargs[name] = _build(f.type, data[name])
        else:
            kwargs[name] = _as_tuples(data[name])
    return cls(**kwargs)
