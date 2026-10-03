"""Run: python tests/test_ppo_runtime.py

Source parity needs the untouched SYMBOL project. Production inference does not.
Only development seeds 30-32 are used; no training or held-out evaluation.
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
SOURCE = HERE.parents[3] / "08.TNSE_02_v.209_SYMBOL_27JUN_114paths_174pools/01.CODE"
DRIVER_NAME = "ppr01_ablation_driver_v2.09.py"
TRAINER_NAME = "ppr01_drl_streamlined_train_ppo_sac_td3_2026_hpc_accel_v2.09.py"
sys.path.insert(0, str(HERE))


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def probe(which):
    # Separate processes keep the checkpoint's stable pickle alias unambiguous.
    driver = import_file("probe_driver", (SOURCE if which == "original" else HERE) / DRIVER_NAME)
    if which == "original":
        sim = driver.import_simulator(SOURCE / TRAINER_NAME)
        args = driver.build_parser().parse_args(["--legacy114-primary-driver-print-plan"])
    else:
        sim = driver.sim
        args = driver.build_parser().parse_args([])
    args.dataset_dir = HERE / "DATASET"
    args.ppo_model = HERE / "ppo_250000_model.zip"
    from stable_baselines3 import PPO
    model = PPO.load(str(args.ppo_model), device="cpu")
    scenarios = driver.canonical_profit13_cohort_scenarios()
    cases = [(s, 30) for s in scenarios] + [(scenarios[1], seed) for seed in (31, 32)]
    output = {"scenarios": [dataclasses.asdict(s) for s in scenarios], "cases": []}
    for scenario, seed in cases:
        clock = Clock()
        env, obs, _ = driver.make_driver_env(sim, args, scenario, seed, "PPO", clock=clock)
        try:
            sim.configure_model_for_env(model, env, "PPO")
            model.predict(obs, deterministic=True)
            obs, _ = sim.reset_env_compat(env, seed=seed)
            row = {"scenario": dataclasses.asdict(scenario), "seed": seed,
                   "initial_state_hash": driver.state_digest(env), "initial_observation": obs.tolist(),
                   "steps": []}
            for _ in range(12):
                env.resume_decision_clock()
                action, _ = model.predict(obs, deterministic=True)
                clock.now += 0.007
                obs, reward, done, info = sim.step_env_compat(env, action)
                row["steps"].append({
                    "action": action.tolist(), "observation": obs.tolist(), "reward": reward,
                    "done": done, "success": bool(info["success"]), "reason": info.get("reason"),
                    "amount_in": info["amount_in"], "amount_fraction": info["amount_fraction"],
                    "profit": env.total_profit_usd, "gas_charged": env.total_gas_usd,
                    "state_hash": driver.state_digest(env),
                })
                if done:
                    break
            row["final_wallet"] = copy.deepcopy(env.current_balance)
            row["final_reserves"] = {key: [v["reserve0"], v["reserve1"]] for key, v in env.current_pools.items()}
            output["cases"].append(row)
        finally:
            env.close()
    print(json.dumps(output, allow_nan=False))


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = import_file("local_driver", HERE / DRIVER_NAME)
        cls.sim = cls.driver.sim
        cls.args = cls.driver.build_parser().parse_args([])
        cls.model = cls.driver.load_model(cls.args.ppo_model)
        cls.scenario = cls.driver.Scenario("development", 2000.0, "2K", 2000, 1.0, 1.0, 4.0)

    def test_01_source_definitions_and_inputs_unchanged(self):
        self.assertEqual(self.sim.file_sha256(SOURCE / TRAINER_NAME), self.sim.SOURCE_TRAINER_SHA256)
        self.assertEqual(self.sim.file_sha256(SOURCE / DRIVER_NAME), self.driver.SOURCE_DRIVER_SHA256)
        original = ast.parse((SOURCE / TRAINER_NAME).read_text())
        port = ast.parse((HERE / "ppo_runtime.py").read_text())
        originals = [n for n in original.body if isinstance(n, ast.FunctionDef)]
        originals += [n for n in ast.walk(original) if isinstance(n, ast.ClassDef)]
        definitions = {ast.dump(n) for n in originals}
        port_definitions = [n for n in port.body if isinstance(n, ast.FunctionDef)]
        port_definitions += [n for n in ast.walk(port) if isinstance(n, ast.ClassDef)]
        checked = 0
        for node in port_definitions:
            self.assertIn(ast.dump(node), definitions, node.name)
            checked += 1
        self.assertGreater(checked, 30)
        files = sorted((HERE / "DATASET").iterdir())
        self.assertEqual(len(files), 18)
        for file in files:
            self.assertEqual(self.sim.file_sha256(file), self.sim.file_sha256(SOURCE / "DATASET" / file.name))
        self.assertEqual(self.sim.file_sha256(self.args.ppo_model), self.driver.MODEL_SHA256)

    def test_02_original_vs_port_all_23_settings(self):
        results = []
        for which in ("original", "port"):
            print(f"Comparing {which}: 23 settings at seed 30, plus seeds 31/32.", flush=True)
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--probe", which],
                cwd=HERE, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True, text=True, timeout=240, check=True,
            )
            results.append(json.loads(process.stdout))
        self.assertEqual(len(results[0]["scenarios"]), 23)
        self.assertEqual(len(results[0]["cases"]), 25)
        self.assertEqual(results[0], results[1])
        steps = [s for case in results[0]["cases"] for s in case["steps"]]
        self.assertTrue(any(s["success"] for s in steps))
        self.assertTrue(any(not s["success"] for s in steps))
        digest = hashlib.sha256(json.dumps(results[0], sort_keys=True).encode()).hexdigest()
        print(f"Exact parity: {len(steps)} decisions, matching actions/observations/rewards/wallet/reserves/gas. SHA256 {digest}", flush=True)

    def test_03_rejection_does_not_mutate_or_stop_at_200(self):
        clock = Clock()
        env, obs = self.driver.prepare_episode(self.args, self.scenario, 30, self.model, clock)
        try:
            before = self.driver.state_digest(env)
            for _ in range(205):
                env.resume_decision_clock()
                obs, reward, terminated, truncated, info = env.step(self.sim.np.array([0.0, 0.0], dtype="float32"))
                self.assertFalse(terminated or truncated)
                self.assertFalse(info["success"])
                self.assertEqual(reward, -0.1)
                self.assertEqual(self.driver.state_digest(env), before)
                self.assertEqual(env.total_gas_usd, 0.0)
            action, _ = self.model.predict(obs, deterministic=True)
            env.resume_decision_clock()
            clock.now += 4.001
            _, reward, terminated, truncated, info = env.step(action)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["stop_reason"], "C6_wall_clock_budget_exhausted")
            self.assertEqual(reward, 0.0)
            self.assertEqual(self.driver.state_digest(env), before)
            self.assertEqual(env.total_gas_usd, 0.0)
        finally:
            env.close()

    def test_04_accepted_swaps_and_c1_rejection(self):
        env, obs = self.driver.prepare_episode(self.args, self.scenario, 30, self.model, Clock())
        try:
            saw_rejection = False
            for _ in range(16):
                before = self.driver.state_snapshot(env)
                before_gas = env.total_gas_usd
                env.resume_decision_clock()
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.sim.step_env_compat(env, action)
                after = self.driver.state_snapshot(env)
                if info["success"]:
                    changed = {k for k in before["pools"] if before["pools"][k] != after["pools"][k]}
                    self.assertEqual(changed, set(info["path"][3]))
                    self.assertAlmostEqual(env.total_gas_usd - before_gas, env.current_gas.gas_cost_usd)
                    self.assertAlmostEqual(env.total_profit_usd, env.total_gross_usd - env.total_gas_usd)
                else:
                    self.assertEqual(info["reason"], "C1_not_profitable_after_gas")
                    self.assertEqual(reward, 0.0)
                    self.assertEqual(before, after)
                    self.assertEqual(before_gas, env.total_gas_usd)
                    saw_rejection = True
                if done:
                    break
            self.assertTrue(saw_rejection)
        finally:
            env.close()

    def test_05_real_c6(self):
        row = self.driver.run_episode(self.args, self.scenario, 30, self.model)
        self.assertTrue(row["episode_valid"])
        self.assertFalse(row["submitted_onchain"])
        self.assertEqual(row["terminal_condition"], "C6_wall_clock_budget_exhausted")
        self.assertGreater(row["profit_usd"], 0.0)
        self.assertGreaterEqual(row["cumulative_decision_time_seconds"], 4.0)
        self.assertEqual(row["executed_transactions"], row["c6_commits_within_budget"])
        # A rejection can already be counted when its final observation crosses C6.
        # In that case the C6 flag describes that same attempt, not an extra one.
        unclassified = row["decision_attempts"] - row["executed_transactions"] - row["screened_rejections"]
        self.assertIn(unclassified, (0, int(row["c6_expired_before_commit"])))
        print("Real C6 smoke: " + json.dumps({k: row[k] for k in (
            "profit_usd", "decision_attempts", "executed_transactions", "terminal_condition")}), flush=True)

    def test_06_bad_inputs_and_no_result_overwrite(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / "result.json"
            self.driver.write_result(path, {"status": "test"})
            with self.assertRaises(FileExistsError):
                self.driver.write_result(path, {"status": "overwrite"})
            self.assertEqual(json.loads(path.read_text()), {"status": "test"})
            with self.assertRaises(ValueError):
                self.driver.write_result(Path(directory) / "invalid.json", {"profit": float("nan")})
            with self.assertRaises(ValueError):
                self.driver.load_model(path)

    def test_07_global_economic_stop_rules(self):
        cases = {
            "C1": "C1_no_profitable_feasible_transaction",
            "C2": "C2_no_funded_start_token",
            "C3": "C3_no_reserve_feasible_path",
        }
        for constraint, expected in cases.items():
            env, obs = self.driver.prepare_episode(self.args, self.scenario, 30, self.model, Clock())
            try:
                action, _ = self.model.predict(obs, deterministic=True)
                if constraint == "C1":
                    env.current_gas = dataclasses.replace(env.current_gas, gas_cost_usd=1e12)
                elif constraint == "C2":
                    env.current_balance = {token: 0.0 for token in env.current_balance}
                else:
                    for pool in env.current_pools.values():
                        pool["reserve0"] = pool["reserve1"] = 0.0
                before = self.driver.state_digest(env)
                _, _, terminated, truncated, info = env.step(action)
                self.assertTrue(terminated, constraint)
                self.assertFalse(truncated, constraint)
                self.assertEqual(info["stop_reason"], expected)
                self.assertEqual(self.driver.state_digest(env), before)
            finally:
                env.close()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--probe":
        probe(sys.argv[2])
    else:
        unittest.main(verbosity=2)
