from typing import Callable, Literal

import numpy as np

from config import EnvConfig, Phase0Config
from env import GREATER, LESS, PAD, Observation

Policy = Callable[[Observation], np.ndarray]
POLICY_NAMES = ("binary", "interpolation", "interpolate_once", "vf_midpoint", "guarded_interpolation",
                "interpolate_k_vf", "adaptive_switch", "probe_extreme_bisect", "probe_extreme_interpolate")
VALUE_BASED = frozenset(POLICY_NAMES) - {"binary"}
ALWAYS_FEASIBLE = frozenset({"interpolation", "vf_midpoint", "guarded_interpolation", "interpolate_k_vf",
                             "adaptive_switch", "probe_extreme_interpolate"})  #every query inside the value-feasible range


def _target(obs: Observation) -> np.ndarray:
    if obs.target is None:
        raise ValueError("value-based teacher needs val feedback; obs.target is None under cmp")
    return obs.target


def comparison_codes(obs: Observation) -> np.ndarray:
    asked = obs.queries != PAD  #history mask by queries: -1 is both the pad and the lower sentinel value
    codes = obs.feedback if obs.target is None else np.sign(obs.feedback - obs.target[:, None]) + 1
    return np.where(asked, codes, PAD)


def bracket(obs: Observation) -> tuple[np.ndarray, np.ndarray]:
    codes = comparison_codes(obs)
    i_left = np.where(codes == LESS, obs.queries, -1).max(axis=1)  #sentinel index -1
    i_right = np.where(codes == GREATER, obs.queries, obs.n[:, None]).min(axis=1)  #sentinel index N
    return i_left + 1, i_right - 1


def value_bounds(obs: Observation, v: int) -> tuple[np.ndarray, np.ndarray]:
    _target(obs)
    codes = comparison_codes(obs)
    a = np.where(codes == LESS, obs.feedback, -1).max(axis=1)  #array is sorted: value at the largest LESS index
    b = np.where(codes == GREATER, obs.feedback, v).min(axis=1)  #sentinel value V
    return a, b


def value_feasible_bracket(obs: Observation, v: int) -> tuple[np.ndarray, np.ndarray]:
    t = _target(obs)
    lo, hi = bracket(obs)
    a, b = value_bounds(obs, v)
    return np.maximum(lo, (hi + 1) - (b - t)), np.minimum(hi, (lo - 1) + (t - a))


def binary_search(obs: Observation) -> np.ndarray:
    lo, hi = bracket(obs)
    return (lo + hi) // 2


def interpolation_search(obs: Observation, v: int) -> np.ndarray:
    target = _target(obs)
    lo, hi = bracket(obs)
    a, b = value_bounds(obs, v)
    m, size, t = hi - lo + 1, b - a - 1, target - a  #m unknowns, L integers strictly between a and b, T's position among them
    num = 2 * (m - 1) * (t - 1) + (size - 1)  #floor(x + 1/2) of x = (m-1)(t-1)/(L-1), in integers
    den = 2 * np.maximum(size - 1, 1)  #L - 1 >= m - 1 >= 1 whenever m > 1
    return lo + np.where(m > 1, num // den, 0)


def interpolate_once(obs: Observation, v: int) -> np.ndarray:
    return np.where(obs.steps == 0, interpolation_search(obs, v), binary_search(obs))


def value_feasible_midpoint(obs: Observation, v: int) -> np.ndarray:
    vlo, vhi = value_feasible_bracket(obs, v)
    return (vlo + vhi) // 2


def guarded_interpolation(obs: Observation, v: int, g: float) -> np.ndarray:
    if not 0.0 <= g < 0.5:
        raise ValueError(f"g={g} must lie in [0, 0.5)")
    vlo, vhi = value_feasible_bracket(obs, v)
    pad = np.floor(g * (vhi - vlo + 1)).astype(np.int64)  #never empties the band for g < 1/2
    return np.clip(interpolation_search(obs, v), vlo + pad, vhi - pad)


def interpolate_k_then_vf_midpoint(obs: Observation, v: int, k: int) -> np.ndarray:
    if k < 1:
        raise ValueError(f"k={k} must be >= 1")
    return np.where(obs.steps < k, interpolation_search(obs, v), value_feasible_midpoint(obs, v))


def observation_prefix(obs: Observation, steps: np.ndarray) -> Observation:
    keep = np.arange(obs.queries.shape[1])[None, :] < steps[:, None]
    return Observation(n=obs.n, target=obs.target, queries=np.where(keep, obs.queries, PAD),
                       feedback=np.where(keep, obs.feedback, PAD), steps=steps, done=obs.done)


def value_feasible_widths(obs: Observation, v: int) -> np.ndarray:
    k_max = obs.queries.shape[1]
    widths = np.empty((len(obs.n), k_max + 1), dtype=np.int64)  #column s: width after the first min(s, steps) queries
    for s in range(k_max + 1):
        vlo, vhi = value_feasible_bracket(observation_prefix(obs, np.minimum(s, obs.steps)), v)
        widths[:, s] = vhi - vlo + 1
    return widths


def adaptive_switch(obs: Observation, v: int, theta: float) -> np.ndarray:
    if not 0.0 < theta <= 1.0:
        raise ValueError(f"theta={theta} must lie in (0, 1]")
    w = value_feasible_widths(obs, v)
    taken = np.arange(obs.queries.shape[1])[None, :] < obs.steps[:, None]
    eliminated = (w[:, :-1] - w[:, 1:]) / w[:, :-1]  #fraction of the range query s was taken in that it removed
    switched = np.any(taken & (eliminated < theta), axis=1)  #permanent after the first sub-threshold query
    return np.where(switched, value_feasible_midpoint(obs, v), interpolation_search(obs, v))


def extreme_rank(n: np.ndarray, t: np.ndarray, v: int) -> np.ndarray:
    return np.where(2 * t < v, np.minimum(n - 1, t), np.maximum(0, n - v + t))  #k_ext of the trap construction


def probe_extreme(obs: Observation, v: int, then: Literal["bisect", "interpolate"]) -> np.ndarray:
    if then not in ("bisect", "interpolate"):
        raise ValueError(f"then={then!r} must be 'bisect' or 'interpolate'")
    first = extreme_rank(obs.n, _target(obs), v)
    rest = binary_search(obs) if then == "bisect" else interpolation_search(obs, v)
    return np.where(obs.steps == 0, first, rest)


def make_policy(name: str, cfg: EnvConfig, ref: Phase0Config = Phase0Config(), k: int = 1) -> Policy:
    if name not in POLICY_NAMES:
        raise ValueError(f"unknown policy {name!r}; expected one of {POLICY_NAMES}")
    if k != 1 and name != "interpolate_k_vf":
        raise ValueError(f"k={k} applies only to interpolate_k_vf, not {name}")
    if name in VALUE_BASED and cfg.feedback != "val":
        raise ValueError(f"{name} needs feedback='val'")
    v = cfg.v
    table: dict[str, Policy] = {
        "binary": binary_search,
        "interpolation": lambda obs: interpolation_search(obs, v),
        "interpolate_once": lambda obs: interpolate_once(obs, v),
        "vf_midpoint": lambda obs: value_feasible_midpoint(obs, v),
        "guarded_interpolation": lambda obs: guarded_interpolation(obs, v, ref.guard_frac),
        "interpolate_k_vf": lambda obs: interpolate_k_then_vf_midpoint(obs, v, k),
        "adaptive_switch": lambda obs: adaptive_switch(obs, v, ref.switch_threshold),
        "probe_extreme_bisect": lambda obs: probe_extreme(obs, v, "bisect"),
        "probe_extreme_interpolate": lambda obs: probe_extreme(obs, v, "interpolate"),
    }
    return table[name]
