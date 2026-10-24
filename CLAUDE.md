# Thesis: interactive search in a hidden sorted array

Read `docs/plan.md` (v1.8.3) before writing or changing anything. It is the specification. If a step here conflicts with the plan, the plan wins; if the plan needs changing, say so and stop.

## How we work

- Plan first. Before writing a file, state in a few lines what it will contain, which plan sections it implements, and what its tests (if any) will check. Wait for approval.
- One file per step. Write a single file, then stop. Arvind reads it, asks questions, and approves before the next file. Do not write two files in one turn.
- File order (plan Section 14 step 1, then onward). Steps 1-3 were written against plan v1.4/v1.5; v1.6 (plan Section 13) requires additive revisions to `config.py`, `teachers.py` and `tests/test_teachers.py` before `dp.py`, each as its own plan-first step:
  1. `config.py` (written; v1.6/v1.8 revision pending: `EnvConfig.cap_mode` with K(N) = N as default, `context_len` 130, `Phase0Config` reference-policy parameters, `CVaRConfig.mean_weight`, `RLConfig.entropy_restart_at_branch`, new `GateConfig`)
  2. `env.py` (written; no v1.6 change)
  3. `teachers.py` (written; v1.6 revision pending: value-feasible midpoint, guarded interpolation, interpolate-k-then-value-feasible-midpoint, adaptive switch) and `tests/test_teachers.py`
  4. `dp.py`
  5. `posterior.py` (mixture rank posterior and the mixture one-step DP reference, plan Section 1)
  6. `bench.py` part 1 (DP, sampling, reference rollouts, Phase 0 projection; plan Section 14 step 2)
  7. `tokenizer.py`
  8. `cvar.py` (shared CVaR estimator, quantile, surrogate weights, diagnostics; plan Appendix B)
  9. `metrics.py` part A (episode summary, per-step rates, DP regret with certification, reference agreement, competence gate)
  10. `phase0.py`
  11. `model.py`
  12. `bench.py` part 2 (network timing on the Phase C-D hardware; sets the Section 3 budgets)
  13. `train_imitation.py`
  14. `eval.py`
  15. `metrics.py` part B (sensitivity, relabelling, disagreement-state and trap-evidence tests, Section 8 decomposition, counterfactual realizability; before any Phase C1 evaluation)
  16. `probes.py` (needed for the Phase A probe pipeline check, plan Section 14 step 4)
  17. `train_rl.py`
- Tests are written immediately after the module they cover, as their own step. Not every module gets one.
- No training runs, no Phase 0 simulation, until the tests for everything they depend on pass and Arvind says go.
- No Section 3 budget is final until `bench.py` has run on the Phase C-D hardware (plan Section 14 step 2).
- Phase D never branches from a checkpoint that failed the competence gate (plan Section 8).
- Record the exact parameter count from the implementation in `config.py` and in `README.md` when `model.py` exists.

## What to test

Test where a bug would be silent and would invalidate every downstream run.

Must test:
- `env.py`: the target value never appears in a `cmp` sequence; cost C = Q + K(N) * 1[failure]; cap K(N) = N under `cap_mode = "linear"` and 2 * ceil(log2 N) + 2 under `"log"`; termination on hit; every sampler yields distinct sorted integers by its exact stated procedure; the environment exposes no bracket.
- `dp.py`: value tables and optimal-action sets match exhaustive enumeration at tiny N, V (e.g. N <= 5, V <= 12) for both `cmp` and `val`/`uniform`; Q of a repeated or out-of-bracket query equals 1 + V(s); every optimal action lies in the value-feasible range.
- `teachers.py`: binary search mean over all targets at N = 64 equals 5.125 exactly; interpolation search always terminates on the target; interpolate-once equals binary search from step 2 onward; value-feasible midpoint always lies in the value-feasible range and equals binary search when that range equals the comparison bracket; guarded interpolation lies in the guard band and eliminates at least floor(g * w) + 1 candidates on either outcome; interpolate-k equals interpolation for the first k steps and the value-feasible midpoint afterwards; the adaptive switch equals interpolation until the first sub-threshold elimination and the value-feasible midpoint afterwards; every policy terminates on the target.
- `posterior.py`: the rank posterior matches brute-force enumeration over arrays at N <= 5, V <= 12 under `trap` and a `trap` mixture; under p = 0 it equals the hypergeometric posterior of Appendix A; the one-step DP policy equals the uniform DP's action set when p = 0; every action lies in the value-feasible range.
- `tokenizer.py`: encode/decode round-trip; `cmp` sequences contain no value tokens.
- `cvar.py`: against brute-force computation on a small hand-made batch with ties at the quantile; check normalization by alpha * B and fractional boundary mass; a batch whose whole tail sits on the quantile atom must give a CVaR value with the correct fractional mass and all-zero surrogate weights, and the strictly-worse count must be 0 for it; alpha = 1 gives the mean. `train_rl.py` uses these functions and does not reimplement them.
- `train_rl.py` data stream: two runs with the same data seed and different action seeds see identical arrays and targets at every update (plan Sections 8, 12).
- `metrics.py`: bracket-elimination fraction, in-bracket rate, in-value-feasible-range rate, and the relabelling test on hand-constructed episodes with known answers; the regret certification flag (plan Section 9): a distinct in-bracket query with m <= r is certified, a repeat with m = r is not, a repeat on the last available query is not; the competence gate on a hand-made evaluation with a known execution-error CVaR share; every counterfactual history used by a test is checked to be realizable by some sorted array of distinct integers in [0, V).

Light or none:
- `model.py`: forward-pass shape check and parameter count; the returned residual stream has shape (B, L, n_layers + 1, d_model) and the logits computed from its last layer equal the forward output.
- `probes.py`: lo/hi labels on hand-built histories equal `teachers.bracket`; the episode-level split puts no episode in both partitions; the last-observation baseline and the random-init baseline run end to end on a tiny checkpoint.
- `train_imitation.py`, `train_rl.py` (beyond the CVaR estimator): a smoke run of a few updates at N <= 8 that completes.
- `eval.py`, `phase0.py`: reference-policy numbers they print must agree with the teacher tests.
- `bench.py`: none beyond running.
- `config.py`: none.

## Code conventions

- Python 3.11+, PyTorch, numpy. Keep dependencies to those plus pytest.
- No emojis anywhere: code, comments, docs, commit messages.
- Comments: inline only, no space after the comment marker (`#comment`). No comment blocks before functions or classes, no docstrings that restate the signature. A function's name and type hints carry its meaning.
- Type hints on every public function.
- Deterministic by construction: every sampler takes an explicit `numpy.random.Generator`; no module-level RNG state.
- `metrics.py` imports `env.py`, `teachers.py`, `dp.py` and `cvar.py` only. Trainers never import `metrics.py`; trainers and `metrics.py` both import `cvar.py`, which is the only CVaR implementation. Nothing in `metrics.py` is ever used to compute a reward.
- The environment never tracks the feasible bracket for the model (plan Section 1).
- K(N) = N is the main setting. A failure is an execution error (a repeat or out-of-bracket query run to the cap), never a slow search; the logarithmic cap exists only for the plan Section 11.5 variant.
- `dp.py` is uncapped (plan Appendix A). Exactness under the cap is a property certified per state by `metrics.py`, not something the DP tracks.
- Observed bounds are a = A[lo - 1] and b = A[hi + 1] with sentinels, never A[lo] or A[hi]; probes, patching and metrics use lo, hi, a, b and rho = (T - a) / (b - a) (plan Sections 1, 10).
- Decision position t is the position of f_{t-1} (of `[T or PAD]` for t = 1), where the logits for q_t are read. Losses, probes and patching act there, never at the q_t token (plan Section 2).
- Trainers keep separate `numpy.random.Generator`s (and torch generators) for environment data and for action sampling. The data generator's seed is shared across compared conditions; the action generator's is per run (plan Sections 8, 12).
- `model.py` exposes the residual stream through its own return value, never through hooks attached by callers; no probe loss is ever added to a search model's training loss (plan Section 10).
- Probe datasets are extracted after training by replaying histories through a frozen checkpoint; training never stores activations. They include off-policy valid histories and are split by whole episode (plan Section 10).
- `train_rl.py` logs the plan Section 3 CVaR diagnostics every update: quantile, strictly-worse count, zero-weight batch fraction, failure mass, CVaR-to-entropy gradient-norm ratio. Any nonzero `cvar.mean_weight` is written into the run's objective label.
- In `cmp` mode the target position holds the dedicated `T_MASK` token; the model must never receive the target value (plan Section 1). `PAD` is right-padding only. T and `val` feedback share the value tokens; the vocabulary is fixed from `n_index_logits` and V, never from a run's `n_min`/`n_max` (plan Section 2).
- All Section 1-4 settings live in `config.py` as a top-level dataclass composed of section sub-dataclasses (env, model, optim, rl, cvar, budget, eval, phase0, gate), with the plan's values as defaults. Fixed validation/test data seeds live there too. Trainers write the resolved config as JSON to the run directory.
- `runs/` is git-ignored.

## Git

- Do not commit, push, branch, or stage unless Arvind asks in that turn.
- Commit messages: imperative, one line, no attribution lines unless asked.

## When unsure

Ask. A wrong assumption baked into `env.py` or `tokenizer.py` means retraining every model.
