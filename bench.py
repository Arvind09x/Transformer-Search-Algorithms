import argparse
import json
import os
import platform
import resource
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from config import EnvConfig, Phase0Config
from dp import ValTable, dp_policy, load_val_table, save_val_table, val_uniform_table
from env import SearchEnv, rollout, sample_batch
from posterior import mixture_one_step_policy
from teachers import Policy, make_policy

FULL = dict(v=256, ns=(32, 64), dp_sizes=((8, 32), (16, 64), (32, 128), (48, 192), (64, 256)), batches=(256, 1000),
            episodes=5000, sample_batches=5)
QUICK = dict(v=64, ns=(8, 16), dp_sizes=((4, 16), (8, 32), (16, 64)), batches=(64, 256), episodes=256, sample_batches=2)
DISTRIBUTIONS = {
    "uniform": dict(distribution="uniform"),
    "adversarial_trap": dict(distribution="adversarial", adversarial="trap"),
    "mixture_trap": dict(distribution="mixture", adversarial="trap", p=0.1),
    "adversarial_clustered": dict(distribution="adversarial", adversarial="clustered"),
    "mixture_clustered": dict(distribution="mixture", adversarial="clustered", p=0.1),
}
BATCH_SWEEP = ("uniform", "mixture_trap")  #measured at every batch size; the rest at the largest only
TRAP_ONLY = ("probe_extreme_bisect", "probe_extreme_interpolate", "mixture_one_step")


def max_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)  #bytes on macOS, KiB on Linux; process high-water mark


def hardware() -> dict[str, Any]:
    import torch
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        ram = None
    return dict(platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                cpu_count=os.cpu_count(), ram_bytes=ram, python=platform.python_version(), numpy=np.__version__,
                torch=torch.__version__, cuda=torch.cuda.is_available(),
                cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                mps=torch.backends.mps.is_available())


def bench_dp(sizes: tuple[tuple[int, int], ...], max_dp: tuple[int, int] | None) -> tuple[list[dict], ValTable | None]:
    rows, table = [], None
    for n_max, l_max in sizes:
        if max_dp is not None and (n_max > max_dp[0] or l_max > max_dp[1]):
            break
        start = time.perf_counter()
        table = val_uniform_table(n_max, l_max)  #timed untraced: tracemalloc slows the build by about 1.5x
        seconds = time.perf_counter() - start
        rss = max_rss_bytes()
        tracemalloc.start()
        val_uniform_table(n_max, l_max)
        _, peak = tracemalloc.get_traced_memory()  #numpy allocations are traced; not everything is, hence max RSS too
        tracemalloc.stop()
        rows.append(dict(n_max=n_max, l_max=l_max, seconds=seconds, traced_peak_bytes=peak, max_rss_bytes=rss))
        print(f"dp  n_max={n_max:3d} l_max={l_max:3d}  {seconds:9.2f} s  traced peak {peak / 2**20:8.1f} MiB  "
              f"max rss {rows[-1]['max_rss_bytes'] / 2**20:8.1f} MiB", flush=True)
    return rows, table


def _cfg(v: int, n_min: int, n_max: int, dist: str, sigma: float = 8.0) -> EnvConfig:
    return EnvConfig(v=v, n_min=n_min, n_max=n_max, feedback="val", clustered_sigma=sigma, **DISTRIBUTIONS[dist])


def bench_sampling(v: int, ns: tuple[int, ...], batches: int, seed: int) -> list[dict]:
    rows = []
    for n in ns:
        for n_min, label in ((n, "fixed"), (4, "variable")):
            for dist in DISTRIBUTIONS:
                if label == "variable" and dist not in BATCH_SWEEP:
                    continue
                cfg = _cfg(v, min(n_min, n), n, dist)
                rng = np.random.default_rng(seed)
                start = time.perf_counter()
                for _ in range(batches):
                    sample_batch(cfg, 1000, rng)
                seconds = time.perf_counter() - start
                rows.append(dict(n=n, n_regime=label, distribution=dist, episodes=1000 * batches, seconds=seconds,
                                 episodes_per_s=1000 * batches / seconds))
    return rows


def _policies(cfg: EnvConfig, dist: str, table: ValTable | None, ref: Phase0Config) -> dict[str, Callable[[], Policy]]:
    out: dict[str, Callable[[], Policy]] = {
        name: (lambda name=name: make_policy(name, cfg, ref))
        for name in ("binary", "vf_midpoint", "interpolation", "guarded_interpolation", "interpolate_once",
                     "adaptive_switch")
    }
    for k in ref.interp_k_grid:
        out[f"interpolate_k_vf_{k}"] = lambda k=k: make_policy("interpolate_k_vf", cfg, ref, k=k)
    if table is not None:
        out["dp_uniform"] = lambda: dp_policy(table, cfg.v)
    if "trap" in dist:
        for name in ("probe_extreme_bisect", "probe_extreme_interpolate"):
            out[name] = lambda name=name: make_policy(name, cfg, ref)
        if table is not None:
            out["mixture_one_step"] = lambda: mixture_one_step_policy(cfg, table)
    return out


def time_policy(cfg: EnvConfig, policy: Policy, batch: int, episodes: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    done, seconds, steps, calls, longest = 0, 0.0, 0, 0, 0
    lengths = []
    while done < episodes:
        env = SearchEnv(cfg, sample_batch(cfg, batch, rng))  #sampling timed separately in bench_sampling
        start = time.perf_counter()
        rollout(env, policy)
        seconds += time.perf_counter() - start
        used = env.queries_used
        steps, calls, longest = steps + int(used.sum()), calls + int(used.max()), max(longest, int(used.max()))
        lengths.append(used)
        done += batch
    lengths = np.concatenate(lengths)
    return dict(episodes=done, seconds=seconds, episodes_per_s=done / seconds, steps_per_s=steps / seconds,
                policy_calls=calls, mean_queries=float(lengths.mean()), max_queries=longest)


def bench_rollouts(v: int, ns: tuple[int, ...], batches: tuple[int, ...], episodes: int, table: ValTable | None,
                   ref: Phase0Config, seed: int) -> list[dict]:
    rows = []
    for n in ns:
        for dist in DISTRIBUTIONS:
            cfg = _cfg(v, n, n, dist)
            sizes = batches if dist in BATCH_SWEEP else batches[-1:]
            for batch in sizes:
                for name, make in _policies(cfg, dist, table, ref).items():
                    policy = make()
                    variants = [("", policy)]
                    if name == "mixture_one_step":  #continuation cache empty, then the same policy object again
                        variants = [("_cold", policy), ("_warm", policy)]
                    for suffix, pol in variants:
                        row = dict(n=n, distribution=dist, batch=batch, policy=name + suffix,
                                   **time_policy(cfg, pol, batch, episodes, seed + (suffix == "_warm")))
                        rows.append(row)
                        print(f"roll n={n:3d} {dist:22s} batch={batch:5d} {name + suffix:26s} "
                              f"{row['episodes_per_s']:10.1f} ep/s  {row['steps_per_s']:11.1f} steps/s  "
                              f"mean q {row['mean_queries']:5.2f}", flush=True)
    return rows


def phase0_grid(ref: Phase0Config, ns: tuple[int, ...]) -> list[dict]:
    grid = []
    for n in ns:
        grid.append(dict(n=n, condition="uniform", measured_as="uniform"))
        grid.append(dict(n=n, condition="adversarial trap", measured_as="adversarial_trap"))
        for sigma in ref.clustered_sigma_grid:
            grid.append(dict(n=n, condition=f"adversarial clustered sigma={sigma:g}", measured_as="adversarial_clustered"))
        for p in ref.p_grid:
            grid.append(dict(n=n, condition=f"mixture trap p={p:g}", measured_as="mixture_trap"))
            for sigma in ref.clustered_sigma_grid:
                grid.append(dict(n=n, condition=f"mixture clustered p={p:g} sigma={sigma:g}",
                                 measured_as="mixture_clustered"))
    return grid


def project_phase0(grid: list[dict], rollouts: list[dict], sampling: list[dict], episodes: int) -> dict[str, Any]:
    rate: dict[tuple[int, str, str], float] = {}
    for row in rollouts:  #largest batch measured wins (rows come in increasing batch order); mixture uses the warm rate
        name = row["policy"].removesuffix("_warm")
        if not row["policy"].endswith("_cold"):
            rate[(row["n"], row["distribution"], name)] = row["episodes_per_s"]
    sample_rate = {(r["n"], r["distribution"]): r["episodes_per_s"] for r in sampling if r["n_regime"] == "fixed"}
    policies = sorted({name for _, _, name in rate})
    per_policy: dict[str, float] = {}
    for name in policies:
        total = 0.0
        for cond in grid:
            trap = "trap" in cond["measured_as"]
            if name in TRAP_ONLY and not trap:
                continue
            total += episodes / rate[(cond["n"], cond["measured_as"], name)]
        per_policy[name] = total
    sampling_s = sum(episodes / sample_rate[(c["n"], c["measured_as"])] for c in grid)  #arrays shared by every policy
    total = sampling_s + sum(per_policy.values())
    return dict(episodes_per_condition=episodes, conditions=len(grid), per_policy_seconds=per_policy,
                per_policy_share={k: v / total for k, v in per_policy.items()}, sampling_seconds=sampling_s,
                total_seconds=total, missing=[] if "dp_uniform" in per_policy else ["dp_uniform", "mixture_one_step"],
                note="mixture p and clustered sigma are projected from p = 0.1, sigma = 8 measurements")


def _json_default(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    raise TypeError(f"not serializable: {type(x)}")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Compute pilot part 1: DP, sampling, reference rollouts, Phase 0 time")
    parser.add_argument("--quick", action="store_true", help="small sizes (V = 64, N in {8, 16}); a few seconds")
    parser.add_argument("--max-dp", type=str, default=None, help="largest DP table as N,V; stops the sweep early")
    parser.add_argument("--save-table", action="store_true", help="save the largest DP table built to runs/dp/")
    parser.add_argument("--table", type=str, default=None, help="load a saved DP table instead of using the sweep's")
    parser.add_argument("--episodes", type=int, default=None, help="episodes per rollout measurement")
    parser.add_argument("--out", type=str, default="runs/bench")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    mode = QUICK if args.quick else FULL
    v, ns, batches = mode["v"], mode["ns"], mode["batches"]
    episodes = args.episodes or mode["episodes"]
    ref = Phase0Config()
    max_dp = tuple(int(x) for x in args.max_dp.split(",")) if args.max_dp else None
    results: dict[str, Any] = dict(mode="quick" if args.quick else "full", started=datetime.now().isoformat(),
                                   hardware=hardware(), seed=args.seed)
    print(json.dumps(results["hardware"], indent=2), flush=True)

    results["dp"], table = bench_dp(mode["dp_sizes"], max_dp)
    if args.save_table and table is not None:
        path = Path("runs/dp") / f"val_uniform_{table.n_max}_{table.l_max}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_val_table(table, path)  #stores n_max and l_max next to the arrays
        results["saved_table"] = str(path)
    if args.table:
        table = load_val_table(args.table)  #raises on a shape mismatch
    if table is not None and (table.n_max < max(ns) or table.l_max < v):
        print(f"dp table ({table.n_max}, {table.l_max}) does not cover N={max(ns)}, V={v}: DP policies skipped")
        table = None
    results["dp_policies_measured"] = table is not None

    results["sampling"] = bench_sampling(v, ns, mode["sample_batches"], args.seed)
    results["max_rss_after_sampling_bytes"] = max_rss_bytes()
    results["rollouts"] = bench_rollouts(v, ns, batches, episodes, table, ref, args.seed)
    results["max_rss_after_rollouts_bytes"] = max_rss_bytes()
    results["phase0_projection"] = project_phase0(phase0_grid(ref, ns), results["rollouts"], results["sampling"],
                                                  ref.episodes)
    results["finished"] = datetime.now().isoformat()

    proj = results["phase0_projection"]
    print(f"\nPhase 0 projection: {proj['conditions']} conditions x {proj['episodes_per_condition']} episodes")
    for name, seconds in sorted(proj["per_policy_seconds"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:26s} {seconds / 3600:9.2f} h  {100 * proj['per_policy_share'][name]:5.1f} %")
    print(f"  {'sampling':26s} {proj['sampling_seconds'] / 3600:9.2f} h")
    print(f"  {'total':26s} {proj['total_seconds'] / 3600:9.2f} h  ({proj['note']})")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{results['mode']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(results, indent=2, default=_json_default) + "\n")
    print(f"results written to {path}")
    return results


if __name__ == "__main__":
    main()
