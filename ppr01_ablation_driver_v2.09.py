#!/usr/bin/env python3
"""COSC723 step 1-4: frozen PPO-only inference, not the four-agent experiment.

Use the existing project Conda Python:
    python ppr01_ablation_driver_v2.09.py --dry-run
    python ppr01_ablation_driver_v2.09.py --seed 30 --latency 4

The default is one development episode, not the held-out 40-49 evaluation.
--all-scenarios selects the original 23 settings for the requested seed.
No training, Codex requests, search baselines, or on-chain submission.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

import ppo_runtime as sim
import numpy as np
from stable_baselines3 import PPO

HERE = Path(__file__).resolve().parent
PROTOCOL_ID = sim.PROTOCOL_ID
EVALUATION_PROTOCOL_ID = "cosc723-ppo-only-port-v1"
MODEL_SHA256 = "1534631a492f8300696df0d6adc32f900272163d5a78db708d57e0ad6655978b"
SOURCE_DRIVER_SHA256 = "2b8214fd664fe6755ca3a2d549716236fde5927768a6f2589c405683ed862ae6"


@dataclasses.dataclass(frozen=True)
class Scenario:
    dimension: str
    ablation_value: float
    ablation_label: str
    balance_usd: int
    c3_liquidity_depth: float
    gas_multiplier: float
    c6_latency_budget_seconds: float


def canonical_profit13_cohort_scenarios() -> list[Scenario]:
    """Preserve 23 source labels, including four repeated reference states."""
    return (
        [Scenario("balance", float(b), f"{b // 1000}K", b, 1.0, 1.0, 12.0)
         for b in range(1000, 10001, 1000)]
        + [Scenario("latency", float(t), f"{t}s", 2000, 1.0, 1.0, float(t))
           for t in (4, 6, 8, 10, 12)]
        + [Scenario("gas", float(g), f"{g}x", 2000, 1.0, float(g), 12.0)
           for g in (1, 2, 4, 8)]
        + [Scenario("liquidity", q, f"{q:g}", 2000, q, 1.0, 12.0)
           for q in (0.25, 0.5, 0.75, 1.0)]
    )


def force_wallet_balance(env, balance_usd: int) -> str:
    desired = f".{int(balance_usd)}.json"
    exact = [(name, wallet) for name, wallet in env.wallets if str(name).endswith(desired)]
    if not exact:
        pattern = re.compile(rf"(?:^|\.){int(balance_usd)}\.json$")
        exact = [(name, wallet) for name, wallet in env.wallets if pattern.search(str(name))]
    if not exact:
        raise ValueError(f"No wallet file for balance {balance_usd}.")
    env.wallets = [exact[0]]
    env.reset_counter = 0
    env.wallet_start_tokens = sorted({token for _, wallet in env.wallets for token, amount in wallet.items() if float(amount) > 0.0})
    return str(exact[0][0])


def scale_liquidity_depth(env, c3_liquidity_depth: float) -> None:
    c3 = float(c3_liquidity_depth)
    if not math.isfinite(c3) or c3 <= 0:
        raise ValueError("C3 liquidity depth must be finite and positive.")
    env.liquidity_depth_multiplier = c3
    env.pool_list = [dataclasses.replace(p, reserve0=p.reserve0 * c3, reserve1=p.reserve1 * c3)
                     for p in copy.deepcopy(env.pool_list)]
    env.pool_template = {pool.pool_id: pool for pool in env.pool_list}
    env.current_pools = env._pool_state_from_template()


def make_driver_env(sim, args, scenario: Scenario, seed: int, method="PPO", clock=None):
    if method != "PPO":
        raise ValueError("This COSC723 runner supports PPO only.")
    env_args = argparse.Namespace(
        dataset_dir=str(args.dataset_dir), amount_min=0.0, amount_max=1.0,
        gas_multiplier=scenario.gas_multiplier, max_paths=5000,
        block_time_seconds=scenario.c6_latency_budget_seconds,
        reward_reference_usd=10000.0, action_mode="auto",
        amount_grid=sim.DEFAULT_AMOUNT_GRID_TEXT, protocol_id=PROTOCOL_ID,
        c1_rejection_reward=0.0, c1_counterfactual_reward=False,
    )
    env = sim.make_env(env_args, seed=seed, wallet_mode="cycle", algo="PPO")
    try:
        if clock is not None:
            env._clock = clock
        force_wallet_balance(env, scenario.balance_usd)
        scale_liquidity_depth(env, scenario.c3_liquidity_depth)
        obs, reset_info = sim.reset_env_compat(env, seed=seed)
        return env, obs, reset_info
    except Exception:
        env.close()
        raise


def load_model(model_path: Path):
    # Load only the approved checkpoint, not an arbitrary pickle ZIP.
    if sim.file_sha256(model_path) != MODEL_SHA256:
        raise ValueError("Checkpoint hash differs from the approved seed-20, 250K PPO.")
    model = PPO.load(str(model_path), device="cpu")
    if (model.observation_space.shape != (355,) or model.action_space.shape != (2,)
            or model.num_timesteps != 250000
            or model.policy.path_feature_mode != "profit15_qstar_capacity"
            or model.policy.amount_treatment != "residual_fstar"):
        raise ValueError("Unexpected PPO observation, action, checkpoint or policy metadata.")
    if not all(bool(sim.th.isfinite(p).all()) for p in model.policy.parameters()):
        raise ValueError("PPO contains non-finite parameters.")
    return model


def prepare_episode(args, scenario: Scenario, seed: int, model, clock=None):
    env, obs, _ = make_driver_env(sim, args, scenario, seed, clock=clock)
    try:
        sim.configure_model_for_env(model, env, "PPO")
        model.predict(obs, deterministic=True)
        obs, _ = sim.reset_env_compat(env, seed=seed)
        return env, obs
    except Exception:
        env.close()
        raise


def state_snapshot(env) -> dict:
    gas = env.current_gas
    return {
        "wallet": {k: float(v) for k, v in sorted(env.current_balance.items())},
        "pools": {k: {"reserve0": float(v["reserve0"]), "reserve1": float(v["reserve1"])}
                  for k, v in sorted(env.current_pools.items())},
        "gas": {
            "source_row_index": int(gas.source_row_index),
            "gas_price_gwei": float(gas.gas_price_gwei), "gas_used": float(gas.gas_used),
            "eth_usd": float(gas.eth_usd), "gas_cost_usd": float(gas.gas_cost_usd),
        },
    }


def state_digest(env) -> str:
    return hashlib.sha256(json.dumps(state_snapshot(env), sort_keys=True).encode("utf-8")).hexdigest()


def run_episode(args, scenario: Scenario, seed: int, model) -> dict:
    """Stop on original economic rules or C6, never on an attempt cap."""
    env, obs = prepare_episode(args, scenario, seed, model)
    try:
        initial = state_snapshot(env)
        initial_hash = state_digest(env)
        attempts = successes = commits_within_c6 = 0
        c6_expired_before_commit = False
        first_action = None
        started = time.perf_counter()
        while True:
            if env.cumulative_decision_time_seconds > scenario.c6_latency_budget_seconds:
                stop_reason = "C6_wall_clock_budget_exhausted"
                break
            if env._cached_common_stop_reason:
                stop_reason = str(env._cached_common_stop_reason)
                break
            env.resume_decision_clock()
            action, _ = model.predict(obs, deterministic=True)
            if not np.isfinite(action).all() or not env.action_space.contains(action):
                raise RuntimeError("PPO produced an invalid or non-finite action.")
            obs, reward, done, info = sim.step_env_compat(env, action)
            attempts += 1
            if first_action is None:
                first_action = np.asarray(action).tolist()
            successes += int(bool(info.get("success", False)))
            commits_within_c6 += int(bool(info.get("c6_commit_within_budget", False)))
            c6_expired_before_commit |= bool(info.get("c6_expired_before_commit", False))
            if info.get("submitted_onchain", False):
                raise RuntimeError("Only local simulation is allowed.")
            if not np.isfinite(obs).all() or not math.isfinite(reward):
                raise RuntimeError("Non-finite observation or reward.")
            if done:
                stop_reason = str(info.get("stop_reason") or env._cached_common_stop_reason)
                if not stop_reason:
                    raise RuntimeError("Environment ended without a stop reason.")
                break
        totals = [env.total_profit_usd, env.total_gross_usd, env.total_gas_usd]
        if not all(math.isfinite(x) for x in totals):
            raise RuntimeError("Non-finite episode totals.")
        return {
            **dataclasses.asdict(scenario), "seed": seed, "method": "PPO",
            "episode_valid": True, "simulated_execution": True, "submitted_onchain": False,
            "profit_usd": float(env.total_profit_usd),
            "gross_profit_usd": float(env.total_gross_usd), "gas_charged_usd": float(env.total_gas_usd),
            "decision_attempts": attempts, "executed_transactions": successes,
            "screened_rejections": int(env.failed_actions), "c6_commits_within_budget": commits_within_c6,
            "c6_expired_before_commit": c6_expired_before_commit,
            "terminal_condition": stop_reason, "first_action": first_action,
            "cumulative_decision_time_seconds": float(env.cumulative_decision_time_seconds),
            "episode_wall_time_seconds": time.perf_counter() - started,
            "initial_state_hash": initial_hash, "final_state_hash": state_digest(env),
            "initial_state": initial, "final_state": state_snapshot(env),
            "loaded_pool_count": len(env.pool_list), "structural_path_count": len(env.triangular_paths),
            "participating_pool_count": len(sim.participating_pool_ids(env.triangular_paths)),
            "observation_dim": int(env.observation_space.shape[0]),
            "dataset_mode": env.dataset_mode,
            "legacy_symbol_collision_audit": env.legacy_symbol_collision,
        }
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=HERE / "DATASET")
    parser.add_argument("--ppo-model", type=Path, default=HERE / "ppo_250000_model.zip")
    parser.add_argument("--seed", type=int, default=30, help="Development default; 40-49 reserved for final evaluation.")
    parser.add_argument("--balance", type=int, choices=range(1000, 10001, 1000), default=2000)
    parser.add_argument("--latency", type=float, choices=(4, 6, 8, 10, 12), default=12)
    parser.add_argument("--gas-multiplier", type=float, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--liquidity", type=float, choices=(0.25, 0.5, 0.75, 1), default=1)
    parser.add_argument("--all-scenarios", action="store_true", help="Original 23 settings for this one seed.")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs, model, and shapes; do not trade.")
    parser.add_argument("--output", type=Path, help="New JSON result path; existing files are never overwritten.")
    return parser


def write_result(path: Path, data: dict) -> None:
    """Publish a fully serialized result without overwriting previous evidence."""
    content = json.dumps(data, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".ppo-result-", delete=False) as f:
            temp = Path(f.name)
            f.write(content)
        os.link(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main() -> None:
    args = build_parser().parse_args()
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(args.dataset_dir)
    args.dataset_dir = sim.discover_dataset_dir(args.dataset_dir)
    if args.seed < 0:
        raise ValueError("Seed must be non-negative.")
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    scenarios = canonical_profit13_cohort_scenarios() if args.all_scenarios else [
        Scenario("development", float(args.balance), f"{args.balance // 1000}K", args.balance,
                 args.liquidity, args.gas_multiplier, args.latency)
    ]
    model = load_model(args.ppo_model)
    inputs = {p.name: sim.file_sha256(p) for p in sorted(args.dataset_dir.iterdir()) if p.is_file()}
    result = {
        "status": "preflight_passed" if args.dry_run else "running",
        "scope": "single-agent PPO development port, not final COSC723 evaluation",
        "protocol_id": PROTOCOL_ID, "evaluation_protocol_id": EVALUATION_PROTOCOL_ID,
        "source_trainer_sha256": sim.SOURCE_TRAINER_SHA256,
        "source_driver_sha256": SOURCE_DRIVER_SHA256,
        "runtime_sha256": sim.file_sha256(Path(sim.__file__)), "driver_sha256": sim.file_sha256(Path(__file__)),
        "model_sha256": sim.file_sha256(args.ppo_model), "dataset_hashes": inputs,
        "model_seed": 20, "checkpoint_decisions": model.num_timesteps,
        "canonical_settings": 23, "requested_episodes": len(scenarios),
        "scenarios": [dataclasses.asdict(s) for s in scenarios], "episodes": [],
    }
    for scenario in scenarios:
        if args.dry_run:
            env, obs = prepare_episode(args, scenario, args.seed, model)
            try:
                if obs.shape != (355,) or len(env.triangular_paths) != 114 or not np.isfinite(obs).all():
                    raise RuntimeError("Dataset/model shape check failed.")
            finally:
                env.close()
            continue
        row = run_episode(args, scenario, args.seed, model)
        result["episodes"].append(row)
        print(json.dumps({k: row[k] for k in (
            "seed", "dimension", "ablation_label", "profit_usd", "decision_attempts",
            "executed_transactions", "terminal_condition")}), flush=True)
    if args.dry_run:
        print(f"Preflight OK: PPO, input 355, 174 pools / 24 participating / 114 paths; "
              f"{len(scenarios)} requested episode(s), 23 canonical settings. No trades executed.")
        return
    result["status"] = "ppo_development_run_complete"
    output = args.output or HERE / "runs" / f"ppo_seed{args.seed}_{time.time_ns()}.json"
    write_result(output, result)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
