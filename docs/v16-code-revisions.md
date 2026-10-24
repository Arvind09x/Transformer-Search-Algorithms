# v1.6 code revisions to files already written

Departures from the original brief (step 3, found while writing the tests; the text below now describes the code):
1. Guarded interpolation: added the elimination check required by plan Section 1 and `CLAUDE.md` (each non-hitting query removes at least floor(g w) + 1 candidates, both outcomes exercised); the original brief tested only band membership.
2. Interpolate-k at k = 1: the difference from `interpolate_once` is restricted to rows with `steps >= 1`; at step 0 both interpolate, so the original "exactly where `value_feasible_midpoint != binary_search`" was false.
3. Adaptive switch hand cases: the original row `[3, 17, 40, 41, 90, 120, 200, 250]` gives interpolation == value-feasible midpoint in all three states, so the cases could not tell the branches apart; replaced by a row where they differ in every state.
4. "Ever switched" on `uniform`: counted over the switch states that governed an action. The hitting query has EQUAL feedback, which the bracket ignores, so it eliminates nothing; counted on the final observation every finished episode looks switched (rate 1.0). Counted on actions the rate is about 0.15.

Scope: the additive changes plan v1.6 Section 13 requires to `config.py`, `teachers.py` and `tests/test_teachers.py`. Three steps, one file per step, plan-first per `CLAUDE.md`. `env.py` and `tests/test_env.py` do not change. Existing tests must keep passing except `test_make_policy_dispatch` in `tests/test_teachers.py`, whose name set stops matching `POLICY_NAMES` after step 2 and is extended in step 3. `test_teachers_stay_in_range` does not fail (the new names fall into its `elif name != "binary"` branch, which checks value-feasibility only at step 0, and they pass it), but it under-tests them and is extended in step 3 as well.

Before step 1, read in this order and keep the plan's conventions in mind throughout: `docs/plan.md` (all of it; Sections 1, 4, 8 and 13 are the ones these steps implement), `CLAUDE.md`, this file, then `config.py`, `env.py`, `teachers.py`, `tests/test_teachers.py`, `tests/test_env.py`.

## Step 1: `config.py`

Plan sections: 1 (reference-policy parameters), 3 (`mean_weight`, entropy restart), 4 (bootstrap), 8 (gate), 13.

1. `Phase = Literal["0", "A", "B", "C1", "C2", "D", "E"]` (Phase C is split; the label changes no setting).
2. `Phase0Config` gains, with validation in `__post_init__`:
   - `guard_frac: float = 0.25  #g of guarded interpolation; clamp to [vlo + floor(g w), vhi - floor(g w)]`; require `0.0 <= guard_frac < 0.5`.
   - `interp_k_grid: tuple[int, ...] = (1, 2)  #k of interpolate-k-then-value-feasible-midpoint`; require non-empty, every entry an int >= 1.
   - `switch_threshold: float = 0.5  #theta of the adaptive switch: minimum eliminated fraction of the value-feasible range`; require `0.0 < switch_threshold <= 1.0`.
   - `bootstrap_resamples: int = 1000  #Monte Carlo uncertainty of the reference gaps`; positive int.
3. `CVaRConfig` gains `mean_weight: float = 0.0  #objective (1 - w) CVaR + w mean; 0 is pure CVaR, anything else is a changed objective and is reported as such`; require `0.0 <= mean_weight < 1.0`.
4. `RLConfig` gains `entropy_restart_at_branch: bool = True  #Phase D: entropy schedule restarts at branching over the Phase D budget, identically in both branches`.
5. New frozen dataclass `GateConfig` placed after `Phase0Config`, every field in [0, 1]:
   ```
   failure_rate_frac_of_alpha: float = 0.1  #failure rate must be below this times cvar.alpha
   repeat_rate_max: float = 0.01  #repeated queries, fraction of all queries
   out_of_range_rate_max: float = 0.01  #queries outside the value-feasible range, fraction of all queries
   execution_cvar_share_max: float = 0.25  #execution-error share of CVaR, fraction of G_ref
   sampled_greedy_gap_max: float = 0.25  #sampled minus greedy CVaR, fraction of G_ref
   ```
6. `Config` gains `gate: GateConfig = field(default_factory=GateConfig)` after `phase0`. No new cross-field checks. `_build` stays strict; no saved config exists yet, so nothing migrates.
7. Comment on `n_index_logits` extended: `#fixed head size, independent of the run's n_max; larger N needs a different action representation (plan 11.4)`.

No tests (`CLAUDE.md`: config has none). Sanity: `Config()` constructs; `Config.from_dict(Config().to_dict()) == Config()`.

## Step 2: `teachers.py`

Plan sections: 1 (definitions of the four policies, value-feasible range, bounds a and b), 13.

Add after `interpolate_once`:

```python
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
```

Notes on these bodies:
- `value_feasible_bracket` already raises under `cmp` (via `_target`), so every new policy is rejected under `cmp` without extra checks.
- `interpolation_search` lies inside the value-feasible range for consistent histories (plan Section 1), so the clip in `guarded_interpolation` only ever acts through the guard band.
- Widths are >= 1 on any history the environment can produce (the target is always feasible), so the division in `adaptive_switch` is safe; do not add an epsilon.
- `observation_prefix` keeps `done` as is; brackets do not read it. `PAD` is already imported from `env`.
- A repeated or out-of-range query eliminates nothing and therefore triggers the switch; the policy never makes such a query itself, so this only matters on externally supplied histories.

Registry changes:

```python
POLICY_NAMES = ("binary", "interpolation", "interpolate_once", "vf_midpoint", "guarded_interpolation",
                "interpolate_k_vf", "adaptive_switch", "probe_extreme_bisect", "probe_extreme_interpolate")
VALUE_BASED = frozenset(POLICY_NAMES) - {"binary"}
ALWAYS_FEASIBLE = frozenset({"interpolation", "vf_midpoint", "guarded_interpolation", "interpolate_k_vf",
                             "adaptive_switch", "probe_extreme_interpolate"})  #every query inside the value-feasible range
```

`make_policy` becomes

```python
def make_policy(name: str, cfg: EnvConfig, ref: Phase0Config = Phase0Config(), k: int = 1) -> Policy:
```

with `from config import EnvConfig, Phase0Config`, and three new table entries: `"vf_midpoint": lambda obs: value_feasible_midpoint(obs, v)`, `"guarded_interpolation": lambda obs: guarded_interpolation(obs, v, ref.guard_frac)`, `"interpolate_k_vf": lambda obs: interpolate_k_then_vf_midpoint(obs, v, k)`, `"adaptive_switch": lambda obs: adaptive_switch(obs, v, ref.switch_threshold)`. `k` is used only by `interpolate_k_vf`; `phase0.py` will loop it over `ref.interp_k_grid`. Raise `ValueError` if `k != 1` is passed for any other name, so a silently ignored parameter cannot happen.

`interpolate_once` stays as is (comparison-midpoint continuation); it is a distinct reference and Section 4 reports both.

## Step 3: `tests/test_teachers.py`

Plan Section 13 and `CLAUDE.md` "What to test". Extend, do not rewrite. Keep the helpers (`_histories`, `_open_rows`, `_observe_after`, `_brute`).

1. `test_make_policy_dispatch`: add the four names to `direct` (with `V`, default `Phase0Config()` values g = 0.25, theta = 0.5, k = 1); add `assert VALUE_BASED == set(POLICY_NAMES) - {"binary"}` unchanged; add that `make_policy("binary", cfg, k=2)` raises `ValueError` and `make_policy("interpolate_k_vf", cfg, k=2)` does not.
2. `test_teachers_stay_in_range`: replace the hard-coded `("interpolation", "probe_extreme_interpolate")` with `ALWAYS_FEASIBLE`; the `elif name != "binary"` branch then covers only `interpolate_once` and `probe_extreme_bisect`.
3. `test_teachers_terminate_on_target` and `test_value_based_rejected_under_cmp` pick up the new names automatically; add the four new callables to the `fn` list in the latter.
4. New `test_vf_midpoint_hand_and_equals_binary_when_unnarrowed`:
   - Hand case: rows `[[10, 11, 12, 15, 16]]`, target_idx `[2]`, queries `[0, 4]`: comparison bracket (1, 3), value-feasible range (1, 2); `value_feasible_midpoint` is 1 and `binary_search` is 2.
   - Hand case: rows `[[3, 17, 40, 41, 90, 120, 200, 250]]`, target_idx `[5]`, queries `[3]`: value-feasible range equals the bracket (4, 7); both midpoints are 5.
   - On `_histories` over `uniform`, `mixture_trap`, `clustered` (>= 1000 open rows): action always in `[vlo, vhi]`; equals `binary_search` exactly on rows where `(vlo, vhi) == (lo, hi)`, and the set of rows where they differ is non-empty.
5. New `test_guarded_interpolation_band`:
   - For g in {0.0, 0.25, 0.4} on the same histories: action in `[vlo + floor(g w), vhi - floor(g w)]` with `w = vhi - vlo + 1`; equals `interpolation_search` wherever interpolation already lies in the band; at g = 0.0 equals interpolation everywhere; at g = 0.25 the number of rows where the clamp acted is > 0.
   - Band non-empty on every row (`vlo + pad <= vhi - pad`).
   - Elimination: for each g, take one more guarded query on every history (`_extend` replays the history and steps once); on rows where it misses, the value-feasible width drops by at least `floor(g w) + 1`, and both LESS and GREATER outcomes occur (> 300 each), so "either outcome" is exercised. `floor(g w)` is computed exactly with `Fraction(str(g))`.
   - Rollouts under `SETTINGS["uniform"]` and `SETTINGS["mixture_trap"]`: `env.success.all()`, and every non-hitting step drops the width by at least `w // 4 + 1` (g = 0.25).
   - `guarded_interpolation(obs, V, 0.5)` raises `ValueError`.
6. New `test_interpolate_k_then_vf_midpoint` (parametrize k in {1, 2, 3}): on recorded rollouts plus `_histories`, rows with `steps < k` equal `interpolation_search`, rows with `steps >= k` equal `value_feasible_midpoint`; at k = 1 the action differs from `interpolate_once` exactly on rows with `steps >= 1` and `value_feasible_midpoint != binary_search` (at step 0 both interpolate), and that set is non-empty; k = 0 raises `ValueError`.
7. New `test_value_feasible_widths_matches_prefix_replay`: for each open row of a few `_histories` batches, replay the first s queries with `_observe_after`-style stepping on that single row and compare `vhi - vlo + 1` with `value_feasible_widths(obs, V)[i, s]` for every `s <= steps`; columns beyond `steps` equal the final width. Column 0 is checked against `value_feasible_bracket` on the fresh observation, not against N: under `val` the initial value-feasible range can already be narrower than [0, N) because T itself constrains the rank.
8. New `test_adaptive_switch`:
   - Hand cases on rows `[[12, 49, 102, 108, 110, 119, 126, 240]]`, target_idx `[3]` (T = 108); in each state the test first asserts `interpolation_search != value_feasible_midpoint`, so the case distinguishes the branches:
     - queries `[4]`: A[4] = 110 > 108 narrows the value-feasible range from 8 to 2 (fraction 0.75 >= theta), no switch; action equals `interpolation_search` (3; midpoint 2).
     - queries `[0]`: A[0] = 12 < 108 narrows it from 8 to 7 (fraction 1/8 < theta), switch; action equals `value_feasible_midpoint` (4; interpolation 3).
     - queries `[0, 4]`: the second query removes 5 of 7, but the switch is permanent; action equals `value_feasible_midpoint` (2; interpolation 3).
   - Random histories: compute `switched` by a per-row replay of each prefix (`_switched_by_replay`, exact `Fraction` comparison against theta) and assert the action equals `value_feasible_midpoint` on switched rows and `interpolation_search` otherwise; among rows where the two differ, both switched rows and unswitched rows with `steps >= 1` occur.
   - Rollouts on `uniform`, `mixture_trap`, `clustered`: `env.success.all()`; on `uniform` the fraction of episodes in which some action was taken in the switched state is below 0.5 (measured: about 0.15). It is computed from the recorded action log, not from the final observation: the hitting query has EQUAL feedback, the bracket ignores it, so it eliminates nothing and would mark every finished episode as switched. This is a loose sanity bound, not a derived constant; if it fails, look at the distribution of eliminated fractions before changing the test.
   - `adaptive_switch(obs, V, 0.0)` and `(obs, V, 1.5)` raise `ValueError`.

## Not in these steps

- `metrics.py` (gate and decomposition), `train_rl.py` (RNG separation, CVaR diagnostics, `mean_weight`), `phase0.py` (bootstrap, G_ref), `bench.py`: later steps in the `CLAUDE.md` file order, specified by plan Sections 3, 8, 9, 13, 14.
