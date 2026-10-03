"""Run with only stdlib: python -B -S tests/test_execution_checker.py"""

from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path
import pickle
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from execution_checker import check_trade


class ExecutionCheckerTests(unittest.TestCase):
    def setUp(self):
        self.pools = {
            "ab": {"token0": "A", "token1": "B", "reserve0": 100.0,
                   "reserve1": 100.0, "fee": 0.0, "metadata": {"tags": ["v2"]}},
            "bc": {"token0": "C", "token1": "B", "reserve0": 100.0,
                   "reserve1": 100.0, "fee": 0.0},
            "ca": {"token0": "A", "token1": "C", "reserve0": 300.0,
                   "reserve1": 100.0, "fee": 0.0},
            "unused": {"token0": "D", "token1": "E", "reserve0": 200.0,
                       "reserve1": 400.0, "fee": 0.003},
        }
        self.wallet = {"A": 100.0, "B": 7.0, "C": 9.0, "ETH": 0.0}
        self.path = ("A", "B", "C", ("ab", "bc", "ca"))
        self.prices = {"A": 2.0, "B": 3.0, "C": 4.0, "ETH": 2000.0}

    def check(self, **overrides):
        inputs = dict(pools=self.pools, wallet=self.wallet, path=self.path,
                      amount_in=10.0, prices=self.prices, gas_cost_usd=0.5)
        inputs.update(overrides)
        # Byte comparison handles NaNs, and catches changes even on rejection.
        before = pickle.dumps(inputs)
        result = check_trade(**inputs)
        self.assertEqual(pickle.dumps(inputs), before)
        self.assertIs(type(result["success"]), bool)
        self.assertIsInstance(result["reason"], str)
        self.assertIsInstance(result["constraint"], str)
        return result

    def assert_rejected(self, result, constraint=None, wallet=None):
        self.assertFalse(result["success"])
        self.assertTrue(result["reason"])
        self.assertTrue(result["constraint"])
        if constraint is not None:
            self.assertEqual(result["constraint"], constraint)
        for field in ("gross_profit_usd", "net_profit_usd", "gas_cost_usd",
                      "final_amount", "token_gain"):
            self.assertEqual(result[field], 0.0, field)
        self.assertEqual(result["pool_updates"], {})
        expected_wallet = self.wallet if wallet is None else wallet
        self.assertEqual(pickle.dumps(result["wallet_after"]),
                         pickle.dumps(expected_wallet))
        self.assertIsNot(result["wallet_after"], expected_wallet)

    def assert_close(self, actual, expected):
        self.assertTrue(math.isclose(actual, float(expected), rel_tol=1e-13,
                                     abs_tol=1e-13), (actual, expected))

    def test_zero_fee_hand_calculated_answer_and_reverse_orientations(self):
        result = self.check()
        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "ok")
        self.assertEqual(result["constraint"], "")
        # 10 -> 100/11 -> 25/3 -> 300/13; start-token USD price is 2.
        self.assert_close(result["final_amount"], Fraction(300, 13))
        self.assert_close(result["token_gain"], Fraction(170, 13))
        self.assert_close(result["gross_profit_usd"], Fraction(340, 13))
        self.assert_close(result["net_profit_usd"], Fraction(667, 26))
        self.assertEqual(result["gas_cost_usd"], 0.5)
        self.assertEqual(set(result["pool_updates"]), {"ab", "bc", "ca"})
        expected_trace = [
            ("A", "B", 10, Fraction(100, 11), "ab"),
            ("B", "C", Fraction(100, 11), Fraction(25, 3), "bc"),
            ("C", "A", Fraction(25, 3), Fraction(300, 13), "ca"),
        ]
        for actual, expected in zip(result["trace"], expected_trace):
            self.assertEqual(actual[:2], expected[:2])
            self.assertEqual(actual[4], expected[4])
            self.assert_close(actual[2], expected[2])
            self.assert_close(actual[3], expected[3])
        expected_reserves = {
            "ab": (110, Fraction(1000, 11)),
            "bc": (Fraction(275, 3), Fraction(1200, 11)),
            "ca": (Fraction(3600, 13), Fraction(325, 3)),
        }
        for pool_id, reserves in expected_reserves.items():
            updated = result["pool_updates"][pool_id]
            self.assert_close(updated["reserve0"], reserves[0])
            self.assert_close(updated["reserve1"], reserves[1])
            self.assertNotEqual(updated, self.pools[pool_id])
        self.assert_close(result["wallet_after"]["A"], Fraction(1470, 13))
        for token in ("B", "C", "ETH"):
            self.assertEqual(result["wallet_after"][token], self.wallet[token])

    def test_fee_and_gas_exact_rational_known_answer(self):
        for pool_id, fee in (("ab", 0.003), ("bc", 0.002), ("ca", 0.01)):
            self.pools[pool_id]["fee"] = fee
        result = self.check()
        self.assertTrue(result["success"])
        # Independently reduced rational quotes for fees 3/1000, 2/1000, 1/100.
        outputs = (Fraction(99700, 10997), Fraction(49750300, 5996003),
                   Fraction(14775839100, 648853097))
        for entry, output in zip(result["trace"], outputs):
            self.assert_close(entry[3], output)
        self.assert_close(result["gross_profit_usd"], Fraction(16574616260, 648853097))
        self.assert_close(result["net_profit_usd"], Fraction(32500379423, 1297706194))
        self.assert_close(result["wallet_after"]["A"], 90 + outputs[-1])
        for entry in result["trace"]:
            token_in, _, amount_in, amount_out, pool_id = entry
            original, updated = self.pools[pool_id], result["pool_updates"][pool_id]
            in_key = "reserve0" if original["token0"] == token_in else "reserve1"
            out_key = "reserve1" if in_key == "reserve0" else "reserve0"
            self.assert_close(updated[in_key], original[in_key] + amount_in)
            self.assert_close(updated[out_key], original[out_key] - amount_out)
            self.assertGreater(updated["reserve0"] * updated["reserve1"],
                               original["reserve0"] * original["reserve1"])

    def test_sequential_trades_use_new_reserves_and_wallet(self):
        exact = {key: {"reserve0": Fraction(pool["reserve0"]),
                       "reserve1": Fraction(pool["reserve1"])}
                 for key, pool in self.pools.items()}
        exact_balance = Fraction(100)
        state, wallet = deepcopy(self.pools), deepcopy(self.wallet)
        final_outputs = []
        for _ in range(2):
            amount = Fraction(10)
            for pool_id, in_key, out_key in (("ab", "reserve0", "reserve1"),
                                            ("bc", "reserve1", "reserve0"),
                                            ("ca", "reserve1", "reserve0")):
                pool = exact[pool_id]
                # Solve x_new*y_new=k instead of reusing the checker formula.
                invariant = pool[in_key] * pool[out_key]
                pool[in_key] += amount
                remaining = invariant / pool[in_key]
                amount = pool[out_key] - remaining
                pool[out_key] = remaining
            exact_balance += amount - 10
            result = self.check(pools=state, wallet=wallet)
            self.assertTrue(result["success"])
            self.assert_close(result["final_amount"], amount)
            self.assert_close(result["wallet_after"]["A"], exact_balance)
            for pool_id, pool in result["pool_updates"].items():
                for key in ("reserve0", "reserve1"):
                    self.assert_close(pool[key], exact[pool_id][key])
            state = dict(state, **result["pool_updates"])
            wallet = result["wallet_after"]
            final_outputs.append(result["final_amount"])
        self.assertLess(final_outputs[1], final_outputs[0])
        self.assertEqual(state["unused"], self.pools["unused"])

    def test_third_leg_failure_discards_first_two_staged_swaps(self):
        for failure in ("missing", "disconnected", "reserve", "overflow"):
            with self.subTest(failure=failure):
                pools = deepcopy(self.pools)
                if failure == "missing":
                    del pools["ca"]
                elif failure == "disconnected":
                    pools["ca"]["token0"] = "D"
                elif failure == "reserve":
                    pools["ca"]["reserve0"] = 0.0
                else:
                    pools["ca"]["reserve0"] = 1e308
                result = self.check(pools=pools)
                self.assert_rejected(result)
                self.assertEqual(len(result["trace"]), 2)
                self.assertEqual(result["quoted_gas_cost_usd"], 0.5)

    def test_unprofitable_quotes_have_no_realized_loss_or_gas(self):
        for gas in (0.0, 0.5, 100.0):
            with self.subTest(gas=gas):
                pools = deepcopy(self.pools)
                pools["ca"]["reserve0"] = 100.0
                result = self.check(pools=pools, gas_cost_usd=gas)
                self.assert_rejected(result, "C1")
                self.assertEqual(len(result["trace"]), 3)
                # Symmetric zero-fee cycle: 10 -> 100/11 -> 25/3 -> 100/13.
                self.assert_close(result["counterfactual_final_amount"], Fraction(100, 13))
                self.assert_close(result["counterfactual_gross_profit_usd"], Fraction(-60, 13))
                self.assert_close(result["counterfactual_net_profit_usd_if_submitted"],
                                  Fraction(-60, 13) - Fraction(gas))
                self.assertEqual(result["quoted_gas_cost_usd"], gas)

    def test_gas_alone_can_reject_a_gross_profitable_trade(self):
        result = self.check(gas_cost_usd=30.0)
        self.assert_rejected(result, "C1")
        self.assert_close(result["counterfactual_gross_profit_usd"], Fraction(340, 13))
        self.assert_close(result["counterfactual_net_profit_usd_if_submitted"], Fraction(-50, 13))

    def test_profit_threshold_is_strict_and_net_of_gas(self):
        for minimum, accepted in ((25.0, True), (26.0, False)):
            with self.subTest(minimum=minimum):
                result = self.check(min_profit_usd=minimum)
                self.assertEqual(result["success"], accepted)
                if not accepted:
                    self.assert_rejected(result, "C1")
        pools = deepcopy(self.pools)
        pools["ab"]["reserve1"] = 200.0
        pools["bc"]["reserve0"] = 200.0
        # 100 -> 100 -> 100 -> 150, so net profit is exactly 99.5 USD.
        for minimum, accepted in ((99.49, True), (99.5, False), (99.51, False)):
            with self.subTest(exact_minimum=minimum):
                result = self.check(pools=pools, amount_in=100.0,
                                    min_profit_usd=minimum)
                self.assertEqual(result["success"], accepted)
                if accepted:
                    self.assertEqual(result["net_profit_usd"], 99.5)
                else:
                    self.assert_rejected(result, "C1")
                    self.assertEqual(result["counterfactual_net_profit_usd_if_submitted"], 99.5)
        pools["ca"]["reserve0"] = 200.0
        self.assert_rejected(self.check(pools=pools, amount_in=100.0,
                                        gas_cost_usd=0.0), "C1")

    def test_balance_limit_missing_start_token_and_exact_balance(self):
        self.assert_rejected(self.check(amount_in=101.0), "C2")
        wallet = dict(self.wallet, A=10.0)
        self.assertTrue(self.check(wallet=wallet)["success"])
        self.assert_rejected(self.check(wallet=wallet, amount_in=10.000000000001),
                             "C2", wallet=wallet)
        del wallet["A"]
        self.assert_rejected(self.check(wallet=wallet), "C2", wallet=wallet)

    def test_invalid_amounts_wallet_balances_prices_gas_and_threshold(self):
        nonfinite = (float("nan"), float("inf"), -float("inf"))
        malformed = (None, True, "10", complex(1, 2), 10 ** 400)
        for value in nonfinite + malformed + (0.0, -1.0):
            with self.subTest(amount=value):
                self.assert_rejected(self.check(amount_in=value), "C2")
        for value in nonfinite + malformed + (-1.0,):
            with self.subTest(wallet=value):
                wallet = dict(self.wallet, B=value)
                self.assert_rejected(self.check(wallet=wallet), "C2", wallet=wallet)
            with self.subTest(gas=value):
                self.assert_rejected(self.check(gas_cost_usd=value), "input")
        for value in nonfinite + malformed + (0.0, -1.0):
            with self.subTest(price=value):
                self.assert_rejected(self.check(prices=dict(self.prices, B=value)), "input")
        for value in nonfinite + malformed:
            with self.subTest(minimum=value):
                self.assert_rejected(self.check(min_profit_usd=value), "input")

    def test_invalid_pool_reserves_and_fees(self):
        bad = (None, True, "100", float("nan"), float("inf"), -float("inf"),
               -1.0, 10 ** 400)
        for field in ("reserve0", "reserve1", "fee"):
            invalid = bad + ((0.0,) if field != "fee" else (1.0, 1.01))
            for value in invalid:
                with self.subTest(field=field, value=value):
                    pools = deepcopy(self.pools)
                    pools["ca"][field] = value
                    self.assert_rejected(self.check(pools=pools), "C3")

    def test_invalid_path_shapes_tokens_and_repeated_pools(self):
        paths = (None, "ABC", (), ("A", "B", "C"),
                 ("A", "A", "C", ("ab", "bc", "ca")),
                 ([], "B", "C", ("ab", "bc", "ca")),
                 ("", "B", "C", ("ab", "bc", "ca")),
                 ("A", "B", "C", None), ("A", "B", "C", "abc"),
                 ("A", "B", "C", ("ab", "bc")),
                 ("A", "B", "C", ("ab", "bc", "ca", "unused")),
                 ("A", "B", "C", ("ab", "bc", "ab")),
                 ("A", "B", "C", ("ab", "bc", [])),
                 ("A", "C", "B", ("ab", "bc", "ca")))
        for path in paths:
            with self.subTest(path=path):
                self.assert_rejected(self.check(path=path), "C4")

    def test_invalid_containers_and_missing_pool_fields(self):
        for key in ("pools", "prices"):
            for value in (None, [], 1):
                with self.subTest(key=key, value=value):
                    self.assert_rejected(self.check(**{key: value}))
        for value in (None, [], 1):
            with self.subTest(wallet=value):
                self.assert_rejected(self.check(wallet=value), "C2", wallet={})
        self.assert_rejected(self.check(prices={"B": 2.0}), "input")
        self.assertTrue(self.check(prices={"A": 2.0})["success"])
        for field in ("token0", "token1", "reserve0", "reserve1", "fee"):
            with self.subTest(missing=field):
                pools = deepcopy(self.pools)
                del pools["ca"][field]
                self.assert_rejected(self.check(pools=pools))

    def test_finite_inputs_that_overflow_or_underflow_reject_atomically(self):
        self.assert_rejected(self.check(prices={"A": 1e308}), "input")
        self.assert_rejected(self.check(amount_in=1e-320), "C3")
        pools = deepcopy(self.pools)
        pools["ca"]["reserve0"] = 1e-12
        self.assert_rejected(self.check(pools=pools), "C3")

        pools["ca"]["reserve0"] = 1e294
        wallet = dict(self.wallet, A=sys.float_info.max)
        result = self.check(pools=pools, wallet=wallet)
        self.assert_rejected(result, "input", wallet=wallet)
        self.assertEqual(len(result["trace"]), 3)

        pools["ab"].update(reserve0=1e308, reserve1=1.0, fee=0.5)
        result = self.check(pools=pools, wallet=wallet, amount_in=1e308)
        self.assert_rejected(result, "C3", wallet=wallet)
        self.assertEqual(result["trace"], [])

    def test_returned_state_is_deeply_detached_and_calls_are_repeatable(self):
        before = deepcopy((self.pools, self.wallet, self.prices))
        first, second = self.check(), self.check()
        self.assertEqual(first, second)
        first["pool_updates"]["ab"]["metadata"]["tags"].append("changed")
        first["pool_updates"]["bc"]["reserve0"] = -1
        first["wallet_after"]["A"] = -1
        first["trace"].clear()
        self.assertEqual((self.pools, self.wallet, self.prices), before)
        self.assertEqual(self.check(), second)
        rejected = self.check(gas_cost_usd=100.0)
        rejected["wallet_after"]["B"] = -1
        self.assertEqual((self.pools, self.wallet, self.prices), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
