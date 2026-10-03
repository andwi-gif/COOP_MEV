#!/usr/bin/env python3
"""Step 5-6 development runner: frozen PPO, checked plans, and shared CPMM pools.

No training, on-chain execution, or final seed-40..49 experiment. The original
PPO-only runner remains available for exact source-parity checks.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import heapq
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cosc723_ppo_driver", HERE / "ppr01_ablation_driver_v2.09.py")
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)
sim = driver.sim

from execution_checker import check_trade
from openrouter_plans import validate_plan, rank_candidates, load_bundle

PROTOCOL = "cosc723-shared-arena-development-v1"
CONDITIONS = ("B1", "B2", "B3", "Proposed", "PPO-sharing-no-team-plan")
AGENTS = ("C1", "C2", "A1", "A2")
TOP_K = 8
TRACE_SAMPLE_LIMIT = 32
TIMING = (
    "One shared measured active-wall-clock C6 budget per arena episode. Includes "
    "observations, global economic checks, PPO candidate inference, feasibility "
    "screens, plan ordering, team messaging, queue work, independent execution "
    "checks, commits and online bookkeeping. Model/data loading, warmup, offline "
    "Codex planning, optional rival-attribution diagnostic quotes, final serialization "
    "and reporting are outside C6. Synchronous proposal rounds share one remaining-time "
    "observation and use integer readiness ticks, not fabricated latency seconds."
)


def digest(value):
    return sim.canonical_json_sha256(value)


def public_planning_input(env, scenario):
    """Only public static market/scenario information, never evaluation outcomes."""
    return {
        "schema_version": "coopmev-public-v1",
        "scenario": dataclasses.asdict(scenario),
        "routes": [{"route_index": i, "tokens": [a, b, c, a], "pool_ids": list(ids)}
                   for i, (a, b, c, ids) in enumerate(env.triangular_paths)],
        "pools": [{"pool_id": p.pool_id, "token0": p.token0, "token1": p.token1,
                   "reserve0": p.reserve0, "reserve1": p.reserve1, "fee": p.fee}
                  for p in env.pool_list],
    }


def ppo_candidates(model, env, top_k=TOP_K):
    """One PPO forward pass; every shortlisted route retains its OWN amount head."""
    if type(top_k) is not int or not 1 <= top_k <= len(env.triangular_paths):
        raise ValueError("Invalid PPO shortlist size")
    obs = env._get_observation()
    model.policy.set_training_mode(False)
    with sim.th.no_grad():
        tensor, _ = model.policy.obs_to_tensor(obs)
        distribution = model.policy.get_distribution(tensor)
        logits = distribution.path_distribution.logits[0].cpu().numpy()
        fractions = distribution._fraction_from_logit(distribution.amount_mu)[0].cpu().numpy()
    if not np.isfinite(logits).all() or not np.isfinite(fractions).all():
        raise FloatingPointError("Non-finite PPO candidates")
    order = np.argsort(-logits, kind="stable")[:top_k]
    return [{"route_index": int(i), "fraction": float(fractions[i]), "ppo_rank": rank,
             "amount_in": float(fractions[i]) * env.current_balance.get(env.triangular_paths[i][0], 0.0)}
            for rank, i in enumerate(order)]


def choose_candidate(env, candidates, plan, avoid_route=None):
    feasible = [c for c in candidates if c["amount_in"] > sim.EPS
                and env._simulate_path(env.triangular_paths[c["route_index"]], c["amount_in"])["success"]]
    if not feasible:
        return dict(candidates[0], proposal_feasible=False), False, False
    ordered = rank_candidates(feasible, plan, avoid_route=None) if plan is not None else feasible
    alternatives = [c for c in ordered if c["route_index"] != avoid_route]
    selected = alternatives[0] if avoid_route is not None and alternatives else ordered[0]
    without_plan = [c for c in feasible if c["route_index"] != avoid_route] or feasible
    plan_changed = selected["route_index"] != without_plan[0]["route_index"]
    return dict(selected, proposal_feasible=True), plan_changed, selected["route_index"] != ordered[0]["route_index"]


def validate_message(message, round_id, path_count):
    if (not isinstance(message, dict) or set(message) != {"sender", "round", "route_index"}
            or message["sender"] != "C1" or type(message["round"]) is not int
            or message["round"] != round_id or type(message["route_index"]) is not int
            or not 0 <= message["route_index"] < path_count):
        raise ValueError("Invalid or stale C1 route message")
    return message["route_index"]


def event_priority(agent, agents, round_id, seed):
    # Rotate tie priority without consulting actions, plans, or profits.
    order = [a for a in ("A1", "C1", "A2", "C2") if a in agents]
    shift = (seed + round_id) % len(order)
    order = order[shift:] + order[:shift]
    return order.index(agent)


def assert_same_quote(source, checked):
    if bool(source["success"]) != bool(checked["success"]):
        raise RuntimeError("Source simulator and independent checker disagree on validity")
    if checked["success"]:
        for key in ("final_amount", "gross_profit_usd", "net_profit_usd"):
            if not math.isclose(source[key], checked[key], rel_tol=1e-10, abs_tol=1e-9):
                raise RuntimeError(f"Execution checker disagrees on {key}")
        for pool, after in checked["pool_updates"].items():
            for side in ("reserve0", "reserve1"):
                if not math.isclose(source["pool_state"][pool][side], after[side], rel_tol=1e-10, abs_tol=1e-9):
                    raise RuntimeError("Execution checker disagrees on reserve update")


def run_arena(args, scenario, seed, model, plans, condition, *, clock=time.perf_counter):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    agents = ("C1", "A1") if condition == "B1" else AGENTS
    env, _ = driver.prepare_episode(args, scenario, seed, model)
    try:
        plans = {a: validate_plan(plans[a], a, len(env.triangular_paths)) for a in agents}
        public = public_planning_input(env, scenario)
        wallets = {a: copy.deepcopy(env.current_balance) for a in agents}
        pools = copy.deepcopy(env.current_pools)
        initial = {"wallets": copy.deepcopy(wallets), "pools": copy.deepcopy(pools),
                   "gas": dataclasses.asdict(env.current_gas)}
        scores = {a: {"profit_usd": 0.0, "gross_profit_usd": 0.0, "gas_usd": 0.0,
                      "proposals": 0, "accepted": 0, "rejected": 0, "invalid_proposals": 0,
                      "stale_state_rejections": 0, "deadline_unexecuted": 0} for a in agents}
        totals = {"messages_sent": 0, "messages_used": 0, "plan_changed_route": 0,
                  "duplicate_avoidance_changed_route": 0, "team_duplicate_path_proposals": 0,
                  "team_shared_pool_proposals": 0, "all_shared_pool_proposal_pairs": 0,
                  "rival_consumed_first": 0, "candidate_screens": 0}
        event_sample = []
        phase_seconds = {"proposal_selection": 0.0, "execution_and_queue": 0.0}
        round_id = event_id = 0
        last_stops = {}
        begin = clock()
        diagnostic_seconds = 0.0
        budget = scenario.c6_latency_budget_seconds

        def elapsed():
            return max(0.0, clock() - begin - diagnostic_seconds)

        def bind(agent, observation_time=None):
            env.current_balance = wallets[agent]
            env.current_pools = pools
            env.cumulative_decision_time_seconds = elapsed() if observation_time is None else observation_time

        def quote(proposal):
            return check_trade(pools, wallets[proposal["agent"]], env.triangular_paths[proposal["route_index"]],
                               proposal["amount_in"], env.prices, env.current_gas.gas_cost_usd,
                               min_profit_usd=env.min_profit_usd)

        def trace(row):
            if len(event_sample) < TRACE_SAMPLE_LIMIT:
                event_sample.append(row)

        def feasible_for_attribution(pending):
            nonlocal diagnostic_seconds
            started = clock()
            try:
                return [p for p in pending if quote(p)["success"]]
            finally:
                diagnostic_seconds += max(0.0, clock() - started)

        stop_reason = ""
        while not stop_reason:
            if elapsed() >= budget:
                stop_reason = "C6_wall_clock_budget_exhausted"
                break
            phase = elapsed()
            round_observation_time = phase
            proposals, message = {}, None
            # No commits during proposal construction: all see the same reserves.
            for agent in agents:
                bind(agent, round_observation_time)
                if elapsed() >= budget:
                    break
                economic_stop, _ = env._exact_common_stop_check()
                last_stops[agent] = economic_stop
                if economic_stop:
                    continue
                if elapsed() >= budget:
                    break
                candidates = ppo_candidates(model, env, args.top_k)
                totals["candidate_screens"] += len(candidates)
                avoid = None
                if agent == "C2" and message is not None:
                    received = validate_message(json.loads(json.dumps(message)), round_id, len(env.triangular_paths))
                    if condition in ("Proposed", "PPO-sharing-no-team-plan"):
                        avoid = received
                        totals["messages_used"] += 1
                plan = None if condition == "PPO-sharing-no-team-plan" and agent.startswith("C") else plans[agent]
                candidate, changed, avoided = choose_candidate(env, candidates, plan, avoid)
                totals["plan_changed_route"] += int(changed)
                totals["duplicate_avoidance_changed_route"] += int(avoided)
                proposals[agent] = {**candidate, "agent": agent, "event_id": event_id,
                                    "ready_tick": round_id, "plan_changed_route": changed,
                                    "duplicate_avoided": avoided}
                event_id += 1
                scores[agent]["proposals"] += 1
                scores[agent]["invalid_proposals"] += int(not candidate["proposal_feasible"])
                if agent == "C1" and condition in ("B3", "Proposed", "PPO-sharing-no-team-plan"):
                    message = {"sender": "C1", "round": round_id, "route_index": candidate["route_index"]}
                    validate_message(message, round_id, len(env.triangular_paths))
                    totals["messages_sent"] += 1
            phase_seconds["proposal_selection"] += max(0.0, elapsed() - phase)
            if elapsed() >= budget:
                for a in proposals:
                    scores[a]["deadline_unexecuted"] += 1
                stop_reason = "C6_wall_clock_budget_exhausted"
                break
            if not proposals:
                stop_reason = "global_economic_exhaustion"
                break
            phase = elapsed()
            offered = list(proposals.values())
            for i, left in enumerate(offered):
                for right in offered[i + 1:]:
                    overlap = set(env.triangular_paths[left["route_index"]][3]) & set(env.triangular_paths[right["route_index"]][3])
                    totals["all_shared_pool_proposal_pairs"] += bool(overlap)
                    if {left["agent"], right["agent"]} == {"C1", "C2"}:
                        totals["team_shared_pool_proposals"] += bool(overlap)
                        totals["team_duplicate_path_proposals"] += left["route_index"] == right["route_index"]
            queue = [(p["ready_tick"], event_priority(a, agents, round_id, seed), p["event_id"], p)
                     for a, p in proposals.items()]
            heapq.heapify(queue)
            while queue:
                _, priority, _, proposal = heapq.heappop(queue)
                agent = proposal["agent"]
                if elapsed() >= budget:
                    for p in [proposal] + [x[3] for x in queue]:
                        scores[p["agent"]]["deadline_unexecuted"] += 1
                    stop_reason = "C6_wall_clock_budget_exhausted"
                    break
                bind(agent)
                checked = quote(proposal)
                path = env.triangular_paths[proposal["route_index"]]
                # Recompute latest state, never use a stale candidate-screen result.
                if proposal["amount_in"] > sim.EPS and proposal["amount_in"] <= wallets[agent].get(path[0], 0.0):
                    assert_same_quote(env._simulate_path(path, proposal["amount_in"]), checked)
                threatened = []
                if checked["success"] and agent.startswith("A"):
                    threatened = feasible_for_attribution([item[3] for item in queue if item[3]["agent"].startswith("C")])
                committed_at = elapsed()
                if committed_at >= budget:
                    for p in [proposal] + [x[3] for x in queue]:
                        scores[p["agent"]]["deadline_unexecuted"] += 1
                    stop_reason = "C6_wall_clock_budget_exhausted"
                    break
                score = scores[agent]
                if checked["success"]:
                    # One synchronous commit; no observer runs between these assignments.
                    wallets[agent] = checked["wallet_after"]
                    pools.update(checked["pool_updates"])
                    score["accepted"] += 1
                    for field, source in (("profit_usd", "net_profit_usd"), ("gross_profit_usd", "gross_profit_usd"), ("gas_usd", "gas_cost_usd")):
                        score[field] += checked[source]
                    still_feasible = feasible_for_attribution(threatened) if threatened else []
                    for waiting in threatened:
                        if waiting not in still_feasible:
                            totals["rival_consumed_first"] += 1
                            trace({"kind": "rival_consumed_first", "rival": agent, "affected_agent": waiting["agent"],
                                   "round": round_id, "route_index": waiting["route_index"]})
                else:
                    score["rejected"] += 1
                    score["stale_state_rejections"] += int(proposal["proposal_feasible"])
                trace({"kind": "proposal", **proposal, "priority": priority,
                       "accepted": bool(checked["success"]), "reason": checked["reason"],
                       "realized_profit_usd": checked["net_profit_usd"] if checked["success"] else 0.0,
                       "commit_checked_at_seconds": committed_at})
            phase_seconds["execution_and_queue"] += max(0.0, elapsed() - phase)
            round_id += 1
        active_seconds = elapsed()
        team = sum(s["profit_usd"] for a, s in scores.items() if a.startswith("C"))
        rival = sum(s["profit_usd"] for a, s in scores.items() if a.startswith("A"))
        for a, score in scores.items():
            if score["proposals"] != score["accepted"] + score["rejected"] + score["deadline_unexecuted"]:
                raise RuntimeError("Proposal accounting mismatch")
            gain = sum((wallets[a][t] - initial["wallets"][a][t]) * env.prices.get(t, 0.0) for t in wallets[a])
            if not math.isclose(gain, score["gross_profit_usd"], rel_tol=1e-9, abs_tol=1e-7):
                raise RuntimeError("Wallet/profit accounting mismatch")
        final = {"wallets": wallets, "pools": pools, "gas": dataclasses.asdict(env.current_gas)}
        row = {
            "condition": condition, "scenario": dataclasses.asdict(scenario), "seed": seed,
            "status": "development_episode_complete", "episode_valid": True,
            "submitted_onchain": False, "agents": scores, "team_profit_usd": team,
            "rival_profit_usd": rival, "team_profit_share": team / (team + rival) if team + rival > 0 else 0.0,
            "team_win": team > rival, "invalid_committed_trades": 0,
            **totals, "rounds": round_id, "terminal_condition": stop_reason,
            "last_agent_economic_stop": last_stops, "online_seconds": active_seconds,
            "rival_attribution_diagnostic_seconds": diagnostic_seconds,
            "phase_seconds": phase_seconds, "timing_semantics": TIMING,
            "initial_state_hash": digest(initial), "final_state_hash": digest(final),
            "initial_agent_hashes": {a: digest({"wallet": w, "pools": initial["pools"], "gas": initial["gas"]}) for a, w in initial["wallets"].items()},
            "initial_state": initial, "final_state": final, "public_input_sha256": digest(public),
            "plan_sha256": {a: digest(p) for a, p in plans.items()}, "candidate_count": args.top_k,
            "event_order": "ready_tick, rotating A1/C1/A2/C2 priority by seed+round, event_id",
            "event_sample": event_sample, "event_sample_limit": TRACE_SAMPLE_LIMIT,
            "event_sample_is_complete": sum(s["proposals"] for s in scores.values()) + totals["rival_consumed_first"] <= len(event_sample),
        }
        json.dumps(row, allow_nan=False)
        return row
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(30, 31, 32), default=30)
    parser.add_argument("--scenario-index", type=int, choices=range(23), default=10,
                        help="Default: original latency-4s setting, wallet 2K per agent")
    parser.add_argument("--condition", choices=(*CONDITIONS, "all"), default="B2")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--plans", type=Path, help="Checked plan bundle, bound to the public scenario input")
    parser.add_argument("--fixture-plans", action="store_true", help="Explicit no-LLM test fixture; NEVER reported as Codex output")
    parser.add_argument("--export-planning-input", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.dataset_dir, args.ppo_model = HERE / "DATASET", HERE / "ppo_250000_model.zip"
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    model = driver.load_model(args.ppo_model)
    scenario = driver.canonical_profit13_cohort_scenarios()[args.scenario_index]
    env, _ = driver.prepare_episode(args, scenario, args.seed, model)
    try:
        public = public_planning_input(env, scenario)
        path_count = len(env.triangular_paths)
    finally:
        env.close()
    if args.export_planning_input:
        driver.write_result(args.export_planning_input, public)
        print(f"Public planning input: {args.export_planning_input}")
        return
    if args.dry_run:
        print(json.dumps({"status": "preflight_passed", "scope": "steps5-6 development",
                          "settings": 23, "conditions": list(CONDITIONS), "final_case_count": 1150,
                          "final_run_launched": False, "paths": path_count, "pool_count": 174,
                          "model_sha256": driver.MODEL_SHA256, "timing": TIMING}))
        return
    if bool(args.plans) == bool(args.fixture_plans):
        raise ValueError("Choose either --plans or explicit --fixture-plans")
    if args.fixture_plans:
        plans = {a: {"schema_version": "coopmev-plan-v1", "agent_id": a, "route_preferences": [],
                     "reason": "Development fixture: keep PPO order; no model request."} for a in AGENTS}
        plan_source = "handwritten_fixture_not_codex"
    else:
        plans, plan_source = load_bundle(args.plans, public, path_count)
    conditions = CONDITIONS if args.condition == "all" else (args.condition,)
    result = {"protocol_id": PROTOCOL, "status": "running", "plan_source": plan_source,
              "model_sha256": driver.MODEL_SHA256,
              "source_protocol_id": driver.PROTOCOL_ID, "source_driver_sha256": driver.SOURCE_DRIVER_SHA256,
              "dataset_hashes": {p.name: sim.file_sha256(p) for p in sorted(args.dataset_dir.iterdir()) if p.is_file()},
              "code_hashes": {p.name: sim.file_sha256(p) for p in (Path(__file__), HERE / "execution_checker.py", HERE / "openrouter_plans.py", HERE / "ppo_runtime.py")},
              "rows": []}
    output = args.output or HERE / "runs" / f"arena_dev_seed{args.seed}_{time.time_ns()}.json"
    try:
        for condition in conditions:
            row = run_arena(args, scenario, args.seed, model, plans, condition)
            result["rows"].append(row)
            print(json.dumps({k: row[k] for k in ("condition", "team_profit_usd", "rival_profit_usd", "terminal_condition", "messages_used", "plan_changed_route")}), flush=True)
        matched = [r["initial_state_hash"] for r in result["rows"] if r["condition"] != "B1"]
        if len(set(matched)) > 1:
            raise RuntimeError("Matched conditions started from different wallet/pool/gas states")
        result["status"] = "step56_development_complete"
        driver.write_result(output, result)
        print(f"Saved development evidence: {output}")
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        driver.write_result(output.with_name(output.stem + ".failed.json"), result)
        raise


if __name__ == "__main__":
    main()
