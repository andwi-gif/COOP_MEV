"""Small arena regression tests; no source-project imports or model requests."""
import copy
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coopmev_arena as arena
from execution_checker import check_trade


@dataclass
class Gas:
    gas_cost_usd: float = 1.0
    source_row_index: int = 30


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class Market:
    def __init__(self):
        self.current_balance = {"A": 100.0}
        self.prices = {"A": 1.0, "B": 1.0, "C": 1.0}
        self.current_gas = Gas()
        self.min_profit_usd = 0.0
        self.current_pools = {}
        self.triangular_paths = []
        for family in range(2):
            ids = tuple(f"{family}-{leg}" for leg in range(3))
            self.triangular_paths.append(("A", "B", "C", ids))
            for leg, (a, b) in enumerate((("A", "B"), ("B", "C"), ("C", "A"))):
                self.current_pools[ids[leg]] = dict(pool_id=ids[leg], token0=a, token1=b,
                                                  reserve0=1000.0, reserve1=1400.0 if leg == 2 else 1000.0, fee=0.0)
        self.pool_list = [SimpleNamespace(**p) for p in self.current_pools.values()]

    def _simulate_path(self, path, amount):
        result = check_trade(self.current_pools, self.current_balance, path, amount,
                             self.prices, self.current_gas.gas_cost_usd)
        return {**result, "pool_state": result["pool_updates"]}

    def _exact_common_stop_check(self):
        return None, None

    def close(self):
        pass


class ArenaTests(unittest.TestCase):
    def fixture(self, condition, *, no_profit=False, budget=0.02, quote_cost=0.0):
        market, clock = Market(), Clock()
        self.observation_times = []
        if no_profit:
            for pool in market.current_pools.values():
                pool["reserve1"] = 1000.0
        scenario = arena.driver.Scenario("latency", budget, "fixture", 100.0, 1.0, 1.0, budget)
        plans = {a: dict(schema_version="coopmev-plan-v1", agent_id=a,
                         route_preferences=[], reason="Explicit test fixture") for a in arena.AGENTS}

        def candidates(*_args):
            self.observation_times.append(market.cumulative_decision_time_seconds)
            clock.value += 0.001
            return [dict(route_index=i, fraction=0.6, amount_in=60.0, ppo_rank=i) for i in range(2)]

        def timed_quote(*args, **kwargs):
            clock.value += quote_cost
            return check_trade(*args, **kwargs)

        with patch.object(arena.driver, "prepare_episode", return_value=(market, None)), \
                patch.object(arena, "ppo_candidates", side_effect=candidates), \
                patch.object(arena, "check_trade", side_effect=timed_quote):
            return arena.run_arena(SimpleNamespace(top_k=2), scenario, 32, None,
                                   plans, condition, clock=clock)

    def test_matched_states_and_message_ablation(self):
        rows = {c: self.fixture(c) for c in arena.CONDITIONS}
        self.assertEqual(len({r["initial_state_hash"] for c, r in rows.items() if c != "B1"}), 1)
        self.assertEqual(rows["B2"]["messages_sent"], 0)
        self.assertGreater(rows["B3"]["messages_sent"], 0)
        self.assertEqual(rows["B3"]["messages_used"], 0)
        self.assertGreater(rows["Proposed"]["messages_used"], 0)
        self.assertGreater(rows["Proposed"]["duplicate_avoidance_changed_route"], 0)
        self.assertEqual(rows["B2"]["agents"], rows["B3"]["agents"])
        self.assertEqual(len(rows["B1"]["agents"]), 2)
        self.assertEqual(len(set(self.observation_times[:4])), 1)

    def test_repeatable_rival_consumes_first(self):
        first, second = self.fixture("B2"), self.fixture("B2")
        self.assertEqual(first, second)
        self.assertGreater(first["rival_consumed_first"], 0)
        proposals = [p for p in first["event_sample"] if p["kind"] == "proposal"]
        self.assertEqual(proposals[0]["agent"], "A1")
        self.assertTrue(proposals[0]["accepted"])
        self.assertFalse(proposals[1]["accepted"])
        self.assertTrue(proposals[1]["proposal_feasible"])
        self.assertGreater(first["agents"]["C1"]["stale_state_rejections"], 0)
        self.assertEqual(first["invalid_committed_trades"], 0)
        for score in first["agents"].values():
            self.assertEqual(score["proposals"], score["accepted"] + score["rejected"] + score["deadline_unexecuted"])
        self.assertTrue(all(p["commit_checked_at_seconds"] < 0.02 for p in proposals))

    def test_rejections_do_not_mutate_and_have_no_200_cap(self):
        row = self.fixture("B2", no_profit=True, budget=1.0)
        self.assertGreater(sum(s["rejected"] for s in row["agents"].values()), 200)
        self.assertEqual(row["initial_state_hash"], row["final_state_hash"])
        self.assertEqual(row["terminal_condition"], "C6_wall_clock_budget_exhausted")
        self.assertEqual(row["team_profit_usd"], 0.0)

    def test_selection_keeps_path_specific_amount(self):
        market = Market()
        candidates = [dict(route_index=0, fraction=0.6, amount_in=60., ppo_rank=0),
                      dict(route_index=1, fraction=0.4, amount_in=40., ppo_rank=1)]
        plan = dict(schema_version="coopmev-plan-v1", agent_id="C1", route_preferences=[1], reason="fixture")
        before = copy.deepcopy(candidates)
        chosen, changed, avoided = arena.choose_candidate(market, candidates, plan)
        self.assertEqual(chosen["amount_in"], 40.)
        self.assertTrue(changed)
        self.assertFalse(avoided)
        chosen, changed, avoided = arena.choose_candidate(market, candidates, plan, avoid_route=1)
        self.assertEqual(chosen["amount_in"], 60.)
        self.assertFalse(changed)
        self.assertTrue(avoided)
        self.assertEqual(candidates, before)

    def test_diagnostic_quotes_do_not_prevent_commit_before_c6(self):
        row = self.fixture("B2", budget=0.006, quote_cost=0.001)
        self.assertGreater(row["rival_attribution_diagnostic_seconds"], 0.0)
        self.assertEqual(sum(s["accepted"] for s in row["agents"].values()), 1)
        first = next(p for p in row["event_sample"] if p["kind"] == "proposal")
        self.assertAlmostEqual(first["commit_checked_at_seconds"], 0.005)
        self.assertEqual(row["terminal_condition"], "C6_wall_clock_budget_exhausted")

    def test_message_validation(self):
        good = dict(sender="C1", round=2, route_index=1)
        self.assertEqual(arena.validate_message(good, 2, 114), 1)
        for bad in (dict(good, round=1), dict(good, sender="A1"), dict(good, route_index=True),
                    dict(good, route_index=114), dict(good, extra="secret")):
            with self.assertRaises(ValueError):
                arena.validate_message(bad, 2, 114)

    def test_checker_mismatch_is_fatal(self):
        with self.assertRaises(RuntimeError):
            arena.assert_same_quote({"success": True}, {"success": False})


class FrozenPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = SimpleNamespace(dataset_dir=arena.HERE / "DATASET",
                                   ppo_model=arena.HERE / "ppo_250000_model.zip", top_k=8)
        cls.model = arena.driver.load_model(cls.args.ppo_model)

    def test_top_candidate_matches_original_deterministic_ppo(self):
        scenario = arena.driver.canonical_profit13_cohort_scenarios()[10]
        env, _ = arena.driver.prepare_episode(self.args, scenario, 30, self.model)
        try:
            env.cumulative_decision_time_seconds = 0.0
            action, _ = self.model.predict(env._get_observation(), deterministic=True)
            candidates = arena.ppo_candidates(self.model, env, 114)
            self.assertEqual(candidates[0]["route_index"], int(action[0]))
            self.assertAlmostEqual(candidates[0]["fraction"], float(action[1]), places=7)
            for candidate in candidates:
                path = env.triangular_paths[candidate["route_index"]]
                if candidate["amount_in"] <= arena.sim.EPS:
                    continue
                checked = check_trade(env.current_pools, env.current_balance, path,
                                      candidate["amount_in"], env.prices, env.current_gas.gas_cost_usd)
                arena.assert_same_quote(env._simulate_path(path, candidate["amount_in"]), checked)
        finally:
            env.close()

    def test_public_plan_input_does_not_change_with_development_seed(self):
        from codex_plans import validate_public_input
        for scenario in arena.driver.canonical_profit13_cohort_scenarios():
            public_hashes = []
            for seed in (30, 31):
                env, _ = arena.driver.prepare_episode(self.args, scenario, seed, self.model)
                try:
                    public = arena.public_planning_input(env, scenario)
                    validate_public_input(public)
                    public_hashes.append(arena.digest(public))
                finally:
                    env.close()
            self.assertEqual(public_hashes[0], public_hashes[1], scenario.ablation_label)


if __name__ == "__main__":
    unittest.main(verbosity=2)
