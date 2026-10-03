"""Independent, side-effect-free three-swap CPMM execution check (stdlib only)."""

from collections.abc import Mapping
from copy import deepcopy
import math


_RESERVE_EPS = 1e-12


def _finite_number(value):
    if isinstance(value, (bool, str, bytes)):
        raise ValueError("Expected a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Expected a finite number") from None
    if not math.isfinite(number):
        raise ValueError("Expected a finite number")
    return number


def _identifier(value):
    return isinstance(value, str) and bool(value.strip())


def check_trade(pools, wallet, path, amount_in, prices, gas_cost_usd,
                min_profit_usd=0.0):
    """Return a proposed atomic transition without changing any input.

    Pools are mappings keyed by pool ID, with token0/token1, reserve0/reserve1
    and fee (a fraction in [0, 1)). Path is (A, B, C, (AB, BC, CA)). Only
    these three pools are inspected. Every supplied wallet balance and price
    must be valid; the start token must have a positive USD price.

    Output quotes use fee-adjusted input, but reserves receive the full input.
    Success requires net profit strictly above min_profit_usd. Gas reduces
    USD profit, not the token wallet, matching the runtime's external payer.

    On rejection all realized amounts, including final_amount and gas, are
    zero, pool_updates is empty, and wallet_after is an unchanged copy.
    Trace entries (token_in, token_out, amount_in, amount_out, pool_id) are
    simulated quotes, never committed swaps. A completed rejected quote has
    separate counterfactual fields. Returned mutable state is detached.
    """
    result = {
        "success": False,
        "reason": "",
        "constraint": "",
        "gross_profit_usd": 0.0,
        "net_profit_usd": 0.0,
        "gas_cost_usd": 0.0,
        "final_amount": 0.0,
        "token_gain": 0.0,
        "pool_updates": {},
        "wallet_after": deepcopy(dict(wallet)) if isinstance(wallet, Mapping) else {},
        "trace": [],
    }

    def reject(reason, constraint):
        result["reason"] = reason
        result["constraint"] = constraint
        return result

    try:
        gas = _finite_number(gas_cost_usd)
        minimum = _finite_number(min_profit_usd)
        if gas < 0.0:
            raise ValueError
    except ValueError:
        return reject("invalid_gas_or_min_profit", "input")
    result["quoted_gas_cost_usd"] = gas

    if not isinstance(wallet, Mapping):
        return reject("C2_invalid_wallet", "C2")
    try:
        for token, balance in wallet.items():
            if not _identifier(token) or _finite_number(balance) < 0.0:
                raise ValueError
        initial_amount = _finite_number(amount_in)
        if initial_amount <= 0.0:
            raise ValueError
    except ValueError:
        return reject("C2_invalid_amount_or_wallet", "C2")

    if not isinstance(path, (tuple, list)) or len(path) != 4:
        return reject("C4_invalid_path", "C4")
    token_a, token_b, token_c, pool_ids = path
    tokens = (token_a, token_b, token_c)
    if not all(_identifier(token) for token in tokens) or len(set(tokens)) != 3:
        return reject("C4_invalid_path_tokens", "C4")
    if (not isinstance(pool_ids, (tuple, list)) or len(pool_ids) != 3
            or not all(_identifier(pool_id) for pool_id in pool_ids)
            or len(set(pool_ids)) != 3):
        return reject("C4_three_distinct_pools_required", "C4")

    start_balance = _finite_number(wallet.get(token_a, 0.0))
    if initial_amount > start_balance:
        return reject("C2_insufficient_balance", "C2")

    if not isinstance(prices, Mapping) or token_a not in prices:
        return reject("invalid_prices", "input")
    try:
        for token, price in prices.items():
            if not _identifier(token) or _finite_number(price) <= 0.0:
                raise ValueError
        start_price = _finite_number(prices[token_a])
    except ValueError:
        return reject("invalid_prices", "input")
    if not isinstance(pools, Mapping):
        return reject("C3_invalid_pools", "C3")

    staged_pools = {}
    expected = tokens + (token_a,)
    amount = initial_amount
    for leg, pool_id in enumerate(pool_ids):
        pool = pools.get(pool_id)
        if not isinstance(pool, Mapping):
            return reject("C3_missing_or_invalid_pool", "C3")
        token_in, token_out = expected[leg:leg + 2]
        if pool.get("token0") == token_in and pool.get("token1") == token_out:
            in_key, out_key = "reserve0", "reserve1"
        elif pool.get("token1") == token_in and pool.get("token0") == token_out:
            in_key, out_key = "reserve1", "reserve0"
        else:
            return reject("C4_path_continuity", "C4")
        try:
            reserve_in = _finite_number(pool.get(in_key))
            reserve_out = _finite_number(pool.get(out_key))
            fee = _finite_number(pool.get("fee"))
            if reserve_in <= 0.0 or reserve_out <= 0.0 or not 0.0 <= fee < 1.0:
                raise ValueError
        except ValueError:
            return reject("C3_invalid_reserves_or_fee", "C3")

        effective_input = amount * (1.0 - fee)
        denominator = reserve_in + effective_input
        numerator = effective_input * reserve_out
        if (effective_input <= 0.0 or not math.isfinite(denominator)
                or not math.isfinite(numerator)):
            return reject("C3_nonfinite_or_unrepresentable_swap", "C3")
        amount_out = numerator / denominator
        new_reserve_in = reserve_in + amount
        new_reserve_out = reserve_out - amount_out
        if (not math.isfinite(amount_out) or not 0.0 < amount_out < reserve_out
                or not math.isfinite(new_reserve_in)
                or min(new_reserve_in, new_reserve_out) <= _RESERVE_EPS
                or new_reserve_in <= reserve_in or new_reserve_out >= reserve_out):
            return reject("C3_reserve_or_numeric_limit", "C3")

        staged_pool = deepcopy(dict(pool))
        staged_pool[in_key] = new_reserve_in
        staged_pool[out_key] = new_reserve_out
        staged_pools[pool_id] = staged_pool
        result["trace"].append((token_in, token_out, amount, amount_out, pool_id))
        amount = amount_out

    token_gain = amount - initial_amount
    gross_profit = token_gain * start_price
    net_profit = gross_profit - gas
    final_balance = start_balance - initial_amount + amount
    if not all(math.isfinite(value) for value in
               (token_gain, gross_profit, net_profit, final_balance)):
        return reject("nonfinite_profit_or_wallet", "input")
    if net_profit <= minimum:
        result.update({
            "counterfactual_final_amount": amount,
            "counterfactual_token_gain": token_gain,
            "counterfactual_gross_profit_usd": gross_profit,
            "counterfactual_net_profit_usd_if_submitted": net_profit,
        })
        return reject("C1_not_profitable_after_gas", "C1")

    result["wallet_after"][token_a] = final_balance
    result.update({
        "success": True,
        "reason": "ok",
        "gross_profit_usd": gross_profit,
        "net_profit_usd": net_profit,
        "gas_cost_usd": gas,
        "final_amount": amount,
        "token_gain": token_gain,
        "pool_updates": staged_pools,
    })
    return result
