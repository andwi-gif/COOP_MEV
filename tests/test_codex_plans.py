"""Offline tests: python -B tests/test_codex_plans.py.

No runtime/checker imports, no model/network calls, and every temporary write
stays inside 01.CODE. Subprocess and executable discovery are always mocked.
"""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import codex_plans as plans


def plain(preferences=None, agent="C1"):
    return {"schema_version": "coopmev-plan-v1", "agent_id": agent,
            "route_preferences": [2, 0] if preferences is None else preferences,
            "reason": "Prefer distinct public routes; keep PPO amounts."}


def public_input(path_count=3):
    return {
        "schema_version": "coopmev-public-v1",
        "scenario": {"dimension": "balance", "ablation_value": 2000.0,
                     "ablation_label": "2K", "balance_usd": 2000,
                     "c3_liquidity_depth": 1.0, "gas_multiplier": 1.0,
                     "c6_latency_budget_seconds": 12.0},
        "routes": [{"route_index": index, "tokens": ["A", "B", "C", "A"],
                    "pool_ids": ["ab", "bc", "ca"]} for index in range(path_count)],
        "pools": [{"pool_id": pool_id, "token0": token0, "token1": token1,
                   "reserve0": 1000.0, "reserve1": 2000.0, "fee": 0.003}
                  for pool_id, token0, token1 in (("ab", "A", "B"), ("bc", "B", "C"), ("ca", "C", "A"))],
    }


def candidate(route, fraction, rank):
    return {"route_index": route, "fraction": fraction,
            "amount_in": fraction * (1000 + route * 100), "ppo_rank": rank}


def events(*extra):
    return "\n".join(json.dumps(event) for event in (
        {"type": "thread.started", "thread_id": "thread-not-request-id"},
        {"type": "turn.started"}, *extra,
        {"type": "turn.completed", "usage": {
            "input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 20}},
    ))


class ValidationTests(unittest.TestCase):
    def test_plain_schema_and_detached_result(self):
        for agent in plans.AGENT_IDS:
            source = plain([113, 0, 57], agent)
            result = plans.validate_plan(source, agent, 114)
            self.assertIs(type(result), dict)
            self.assertEqual(result, source)
            result["route_preferences"].clear()
            self.assertEqual(source["route_preferences"], [113, 0, 57])
        self.assertEqual(plans.validate_plan(plain([]), "C1", 1)["route_preferences"], [])

    def test_bad_plan_shapes_types_ranges_and_extra_fields(self):
        bad = [None, [], "{}", {**plain(), "extra": 1}, {**plain(), "rationale": "old field"}]
        for key in plain():
            bad.append({k: v for k, v in plain().items() if k != key})
        for value in ("other", 1, True, None):
            bad.append({**plain(), "schema_version": value})
        for value in ("A1", "C3", [], None):
            bad.append({**plain(), "agent_id": value})
        for value in ([0, 0], [True], [1.0], ["1"], [-1], [114], [None], [0, []], (0, 1), {}, None):
            bad.append({**plain(), "route_preferences": value})
        for value in (None, 1, "", "  ", "x" * 401, "private\nthought", "bad\x00text"):
            bad.append({**plain(), "reason": value})
        for value in bad:
            with self.subTest(value=value), self.assertRaises(plans.PlanError):
                plans.validate_plan(value, "C1", 114)

    def test_agent_and_path_count_are_strict(self):
        for count in (0, -1, 115, True, 114.0, "114", None):
            with self.subTest(count=count), self.assertRaises(plans.PlanError):
                plans.validate_plan(plain(), "C1", count)
        for agent in ("C3", None, [], True):
            with self.subTest(agent=agent), self.assertRaises(plans.PlanError):
                plans.validate_plan(plain(), agent, 114)
        with self.assertRaises(plans.PlanError):
            plans.validate_plan(plain([3]), "C1", 3)

    def test_plan_validation_is_not_llm_or_economic_grading(self):
        with patch.object(plans.subprocess, "run") as run:
            self.assertEqual(plans.validate_plan(plain([113]), "C1", 114), plain([113]))
        run.assert_not_called()

    def test_schema_is_fresh_agent_specific_and_closed(self):
        schema = plans.plan_schema("A2", 3)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(plain()))
        self.assertEqual(schema["properties"]["agent_id"]["enum"], ["A2"])
        self.assertEqual(schema["properties"]["route_preferences"]["items"]["maximum"], 2)
        schema["properties"].clear()
        self.assertTrue(plans.plan_schema("A2", 3)["properties"])


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.items = [candidate(0, 0.1, 0), candidate(1, 0.6, 1), candidate(2, 0.25, 2)]

    def test_plan_changes_route_and_keeps_route_specific_fraction_amount(self):
        snapshot = copy.deepcopy(self.items)
        result = plans.rank_candidates(self.items, plain([2, 0]))
        self.assertEqual([item["route_index"] for item in result], [2, 0, 1])
        self.assertEqual(result[0]["fraction"], 0.25)
        self.assertEqual(result[0]["amount_in"], 300)
        self.assertIs(result[0], self.items[2])
        self.assertEqual(self.items, snapshot)

    def test_ties_and_unlisted_routes_keep_input_ppo_order(self):
        self.items[0]["ppo_rank"] = 900  # Input order, not an untrusted label.
        self.items.append(candidate(2, 0.7, 3))
        result = plans.rank_candidates(self.items, plain([2]))
        self.assertEqual(result, [self.items[2], self.items[3], self.items[0], self.items[1]])

    def test_avoidance_precedes_plan_and_keeps_fallback(self):
        result = plans.rank_candidates(self.items, plain([2, 0]), avoid_route=2)
        self.assertEqual([item["route_index"] for item in result], [0, 1, 2])
        self.assertEqual(plans.rank_candidates([self.items[2]], plain([2]), avoid_route=2), [self.items[2]])
        self.assertEqual(plans.rank_candidates(self.items, plain([2]), avoid_route=113)[0], self.items[2])
        self.assertEqual(plans.rank_candidates([], plain(), avoid_route=0), [])

    def test_no_codex_ablation_uses_same_avoidance(self):
        self.assertEqual(plans.rank_candidates(self.items, None), self.items)
        self.assertEqual(plans.rank_candidates(self.items, plain([])), self.items)
        self.assertEqual(plans.rank_candidates(self.items, None, avoid_route=0), self.items[1:] + self.items[:1])

    def test_candidates_require_valid_routes_and_fractions_without_repair(self):
        bad = [{}, {"route_index": 0}, {"fraction": 0.1}, None]
        bad += [{"route_index": r, "fraction": 0.1} for r in (True, 1.0, -1, 114)]
        bad += [{"route_index": 0, "fraction": f} for f in (-0.1, 1.1, True, "0.1", float("nan"), float("inf"))]
        for item in bad:
            with self.subTest(item=item), self.assertRaises(plans.PlanError):
                plans.rank_candidates([item], None)
        for value in (True, -1, 114, 1.0):
            with self.assertRaises(plans.PlanError):
                plans.rank_candidates(self.items, plain(), value)


class PublicInputTests(unittest.TestCase):
    def test_prepare_is_pure_deterministic_and_agent_specific(self):
        source = public_input()
        with patch.object(plans.subprocess, "run") as run:
            request = plans.prepare_request(source, "C1", 3)
            other = plans.prepare_request(dict(reversed(list(source.items()))), "C1", 3)
            a2 = plans.prepare_request(source, "A2", 3)
        run.assert_not_called()
        self.assertEqual(request, other)
        self.assertEqual(request["public_input_sha256"], a2["public_input_sha256"])
        self.assertNotEqual(request["prompt_sha256"], a2["prompt_sha256"])
        request["public_input"]["routes"].clear()
        self.assertEqual(len(source["routes"]), 3)

    def test_private_input_cannot_hide_under_nested_unknown_fields(self):
        for key in ("seed", "observations", "outcomes", "checker", "profit", "context", "metadata"):
            for location in ("top", "scenario", "route", "pool"):
                source = public_input()
                target = {"top": source, "scenario": source["scenario"],
                          "route": source["routes"][0], "pool": source["pools"][0]}[location]
                target[key] = "DO NOT SEND"
                with self.subTest(key=key, location=location), self.assertRaises(plans.PlanError):
                    plans.prepare_request(source, "C1", 3)

    def test_descriptor_types_route_coverage_and_pool_consistency(self):
        mutations = [
            lambda p: p.update(schema_version="bad"),
            lambda p: p["routes"].pop(),
            lambda p: p["routes"][0].update(route_index=True),
            lambda p: p["routes"][0].update(route_index=1),
            lambda p: p["routes"][0].update(tokens=["A", "B", "C", "D"]),
            lambda p: p["routes"][0].update(pool_ids=["absent", "bc", "ca"]),
            lambda p: p["routes"][0].update(pool_ids=["bc", "ab", "ca"]),
            lambda p: p["pools"][0].update(reserve0=float("nan")),
            lambda p: p["pools"][0].update(reserve1=-1),
            lambda p: p["pools"][0].update(fee=True),
            lambda p: p["pools"][0].update(fee=1),
            lambda p: p["pools"][0].update(pool_id="bc"),
            lambda p: p["scenario"].update(balance_usd={"seed": 40}),
            lambda p: p["scenario"].update(gas_multiplier=10**1000),
        ]
        for mutation in mutations:
            source = public_input()
            mutation(source)
            with self.subTest(source=source), self.assertRaises(plans.PlanError):
                plans.validate_public_input(source, 3)


class AuditTests(unittest.TestCase):
    def test_turns_tokens_and_invocations_are_not_actual_requests(self):
        audit = plans.audit_cli_events(events())
        self.assertEqual(audit["cli_invocations"], 1)
        self.assertEqual(audit["completed_turns"], 1)
        self.assertEqual(audit["token_usage"], {"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 20})
        self.assertIsNone(audit["actual_model_requests"])
        self.assertEqual(audit["request_count_status"], "unverifiable")
        self.assertEqual(plans.audit_cli_events(events() + "\n" + events())["completed_turns"], 2)

    def test_reasoning_secrets_and_errors_never_persist(self):
        raw = events(
            {"type": "item.completed", "item": {"type": "reasoning", "text": "PRIVATE_REASONING"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "RAW_MESSAGE"}},
            {"type": "error", "message": "SECRET_TOKEN"},
            {"type": "turn.failed", "error": "PRIVATE_FAILURE"},
        )
        audit = plans.audit_cli_events(raw)
        self.assertEqual(audit["failures"], ["error", "turn.failed"])
        self.assertEqual(audit["token_usage_status"], "reported_partial")
        for secret in ("PRIVATE_REASONING", "RAW_MESSAGE", "SECRET_TOKEN", "PRIVATE_FAILURE"):
            self.assertNotIn(secret, json.dumps(audit))

    def test_partial_malformed_and_tool_events_are_not_successes(self):
        for raw in ("", "{", "null", '{"type":"turn.completed"}',
                    '{"type":"turn.completed","usage":{"input_tokens":true}}',
                    events({"type": "item.completed", "item": {"type": "command_execution", "command": "cat secret"}})):
            with self.subTest(raw=raw):
                audit = plans.audit_cli_events(raw)
                self.assertTrue(audit["failures"])
                self.assertIsNone(audit["actual_model_requests"])
                self.assertEqual(audit["token_usage_status"], "reported_partial")


class FileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix=".test-codex-plans-", dir=HERE)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        patcher = patch.object(plans, "HERE", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def envelope(self):
        return plans.make_envelope(plain(), public_input(), model="pinned-campus-model", path_count=3,
                                   audit=plans.audit_cli_events(events()), generation_seconds=1.25)

    def test_save_load_public_hash_binding_and_no_overwrite(self):
        envelope = self.envelope()
        path = plans.save_plan("plans/development/C1.json", envelope)
        self.assertEqual(plans.load_plan(path, "C1", 3, public_input=public_input()), plain())
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(envelope["metadata"]["scope"], "development")
        self.assertFalse(envelope["metadata"]["final_eligible"])
        with self.assertRaises(FileExistsError):
            plans.save_plan(path, envelope)
        with self.assertRaises(plans.PlanError):
            plans.load_plan(path, "A1", 3)
        changed = public_input()
        changed["scenario"]["balance_usd"] = 3000
        with self.assertRaises(plans.PlanError):
            plans.load_plan(path, "C1", 3, public_input=changed)

    def test_missing_accounting_is_unknown_not_zero_or_one(self):
        envelope = plans.make_envelope(plain(), public_input(), model="pinned-campus-model", path_count=3)
        audit = envelope["metadata"]["accounting"]
        self.assertIsNone(audit["cli_invocations"])
        self.assertIsNone(audit["actual_model_requests"])
        self.assertIsNone(audit["token_usage"])
        self.assertEqual(envelope["metadata"]["model_provenance"], "caller_reported_unverified")

    def test_provenance_tampering_and_false_certification_rejected(self):
        mutations = [
            lambda e: e["plan"].update(route_preferences=[1]),
            lambda e: e["metadata"].update(public_input_sha256="0" * 64),
            lambda e: e["metadata"].update(prompt="different prompt"),
            lambda e: e["metadata"]["schema"].update(additionalProperties=True),
            lambda e: e["metadata"].update(model_provenance="verified"),
            lambda e: e["metadata"].update(isolation_status="verified"),
            lambda e: e["metadata"].update(final_eligible=True),
            lambda e: e["metadata"].update(path_count=True),
            lambda e: e["metadata"].update(code_sha256="invalid"),
            lambda e: e["metadata"].update(created_at_utc="2026-09-18"),
            lambda e: e["metadata"]["accounting"].update(actual_model_requests=1),
            lambda e: e["metadata"]["accounting"].update(request_count_status="verified"),
            lambda e: e["metadata"]["accounting"].update(reasoning="not allowed"),
        ]
        for index, mutation in enumerate(mutations):
            envelope = self.envelope()
            mutation(envelope)
            with self.subTest(index=index), self.assertRaises(plans.PlanError):
                plans.save_plan(f"plans/bad-{index}.json", envelope)
        self.assertFalse((self.root / "plans").exists())

    def test_duplicate_keys_bare_plans_and_nonfinite_json_rejected(self):
        for index, raw in enumerate((json.dumps(plain()), '{"a":1,"a":2}', '{"a":NaN}', "null")):
            path = self.root / f"bad-{index}.json"
            path.write_text(raw, encoding="utf-8")
            with self.assertRaises(plans.PlanError):
                plans.load_plan(path, "C1", 3)

    def test_output_escape_and_symlink_redirect_rejected(self):
        for path in (self.root / "outside.json", "plans/../outside.json", "outside.json"):
            with self.subTest(path=path), self.assertRaises(plans.PlanError):
                plans.save_plan(path, self.envelope())
        outside = self.root / "not-plans"
        outside.mkdir()
        (self.root / "plans").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(plans.PlanError):
            plans.save_plan("plans/escape.json", self.envelope())
        self.assertEqual(list(outside.iterdir()), [])

    def test_final_always_fails_closed_no_self_asserted_budget(self):
        with patch.object(plans.subprocess, "run") as run:
            for count in (None, 0, 1, 92, 93):
                with self.assertRaises(plans.FinalBudgetError):
                    plans.require_final_budget(actual_model_requests=count, verified=True)
            with self.assertRaises(plans.FinalBudgetError):
                plans.generate_plan(public_input(), "C1", model="pinned-campus-model", path_count=3, final=True)
            with self.assertRaises(plans.FinalBudgetError):
                plans.load_plan(self.root / "missing.json", "C1", 3, final=True)
        run.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_mock_cli_probes_fresh_home_sanitized_environment_and_no_model_dispatch(self):
        observed_homes = []

        def fake_run(command, **kwargs):
            home = Path(kwargs["cwd"])
            observed_homes.append(home)
            self.assertTrue(home.is_relative_to(self.root / "plans"))
            self.assertFalse((home / ".codex" / "auth.json").exists())
            self.assertFalse((home / ".codex" / "config.toml").exists())
            self.assertEqual(kwargs["env"]["HOME"], str(home))
            self.assertEqual(kwargs["env"]["CODEX_HOME"], str(home / ".codex"))
            self.assertEqual(kwargs["env"]["TMPDIR"], str(home))
            self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
            self.assertNotIn("CODEX_API_KEY", kwargs["env"])
            self.assertNotIn("PYTHONPATH", kwargs["env"])
            self.assertNotIn("HTTP_PROXY", kwargs["env"])
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
            self.assertNotIn("input", kwargs)
            if "exec" in command:
                self.assertEqual(command[1:], ["exec", "--ignore-user-config", "--help"])
            else:
                self.assertEqual(command[1:], ["--version"])
            return subprocess.CompletedProcess(command, 0, "codex-cli 0.154.0\n", "SECRET_STDERR")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret-key", "CODEX_API_KEY": "secret-key",
                                     "HTTP_PROXY": "secret-proxy", "PYTHONPATH": "untrusted"}), \
                patch.object(plans.shutil, "which", return_value="/usr/bin/codex"), \
                patch.object(plans.subprocess, "run", side_effect=fake_run) as run:
            for _ in range(2):
                with self.assertRaises(plans.GenerationBlocked) as caught:
                    plans.generate_plan(public_input(), "C1", model="pinned-campus-model", path_count=3)
                audit = json.loads(caught.exception.audit_path.read_text())
                self.assertEqual(audit["accounting"]["actual_model_requests"], 0)
                self.assertEqual(audit["accounting"]["cli_invocations"], 0)
                self.assertEqual(audit["accounting"]["request_count_status"], "verified_not_dispatched")
                self.assertEqual(audit["status"], "blocked")
                self.assertEqual(audit["probe_results"][0]["version"], "0.154.0")
                self.assertNotIn("SECRET_STDERR", json.dumps(audit))
        self.assertEqual(run.call_count, 4)
        self.assertNotEqual(observed_homes[0], observed_homes[2])
        self.assertTrue(all(not home.exists() for home in observed_homes))
        self.assertEqual(len(list((self.root / "plans").glob("blocked-*.json"))), 2)

    def test_probe_failure_timeout_and_missing_cli_remain_audited_and_blocked(self):
        for failure in (subprocess.TimeoutExpired("codex", 15, output="SECRET"), OSError("SECRET")):
            with patch.object(plans.shutil, "which", return_value="/usr/bin/codex"), \
                    patch.object(plans.subprocess, "run", side_effect=failure):
                with self.assertRaises(plans.GenerationBlocked) as caught:
                    plans.generate_plan(public_input(), "C1", model="pinned-campus-model", path_count=3)
            text = caught.exception.audit_path.read_text()
            self.assertNotIn("SECRET", text)
            self.assertTrue(all(probe["status"] == "failed" for probe in json.loads(text)["probe_results"]))
        with patch.object(plans.shutil, "which", return_value=None), patch.object(plans.subprocess, "run") as run:
            with self.assertRaises(plans.GenerationBlocked) as caught:
                plans.generate_plan(public_input(), "C1", model="pinned-campus-model", path_count=3)
        run.assert_not_called()
        self.assertEqual(json.loads(caught.exception.audit_path.read_text())["probe_results"][0]["status"], "unavailable")

    def test_cli_reports_block_without_secrets(self):
        path = self.root / "public.json"
        path.write_text(json.dumps(public_input()), encoding="utf-8")
        output = io.StringIO()
        with patch.object(plans.shutil, "which", return_value=None), \
                patch.object(plans.subprocess, "run") as run, redirect_stdout(output):
            status = plans.main(["--public-input", str(path), "--agent-id", "C1", "--path-count", "3",
                                 "--model", "pinned-campus-model"])
        self.assertEqual(status, 2)
        self.assertIn("Blocked before model dispatch", output.getvalue())
        run.assert_not_called()

    def fake_auth(self):
        path = self.root / "cached-auth.json"
        path.write_text(json.dumps({"auth_mode": "chatgpt", "OPENAI_API_KEY": None,
                                   "tokens": {"access_token": "SECRET_ACCESS", "refresh_token": "SECRET_REFRESH",
                                              "id_token": "SECRET_ID", "account_id": "SECRET_ACCOUNT"},
                                   "last_refresh": "2026-09-18T00:00:00Z"}))
        return path

    def live_runner(self, *, failure=None, contaminated=False):
        def run(command, **kwargs):
            home = Path(kwargs["env"]["CODEX_HOME"])
            self.assertTrue(home.is_relative_to(self.root / "plans"))
            self.assertTrue(home.is_dir())
            if command[1] == "--version":
                return subprocess.CompletedProcess(command, 0, "codex-cli 0.154.0\n", "")
            if command[1] == "debug":
                self.assertFalse((home / "auth.json").exists())
                messages = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": command[-1]}]}]
                if contaminated:
                    messages.append({"role": "developer", "content": "PRIVATE_PROJECT_CONTEXT"})
                return subprocess.CompletedProcess(command, 0, json.dumps(messages), "")
            if command[1] == "sandbox":
                allowed = command[-2:] == ["/bin/cat", str(Path(kwargs["cwd"]) / "public-canary.txt")]
                return subprocess.CompletedProcess(command, 0 if allowed else 1,
                                                   "PUBLIC_CANARY" if allowed else "", "")
            self.assertEqual(command[1], "exec")
            self.assertEqual(command[-1], "-")
            self.assertIn("--ignore-user-config", command)
            self.assertIn("--ignore-rules", command)
            self.assertIn("--ephemeral", command)
            self.assertIn("features.shell_tool=false", command)
            self.assertIn("features.view_image=false", command)
            self.assertIn("features.apply_patch_freeform=false", command)
            self.assertIn('forced_login_method="chatgpt"', command)
            self.assertIn('default_permissions="planner"', command)
            self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
            self.assertNotIn("CODEX_API_KEY", kwargs["env"])
            self.assertEqual((home / "auth.json").stat().st_mode & 0o777, 0o600)
            self.assertFalse((home / "auth.json").is_symlink())
            self.assertTrue((Path(kwargs["cwd"]) / ".git").is_dir())
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, 1, output=events().encode())
            output = Path(command[command.index("--output-last-message") + 1])
            output.write_text("{}" if failure == "malformed" else json.dumps(plain()))
            return subprocess.CompletedProcess(command, 1 if failure == "exit" else 0,
                events({"type": "item.completed", "item": {"type": "reasoning", "text": "PRIVATE_REASONING"}}),
                "SECRET_STDERR")
        return run

    def test_live_development_success_is_saved_but_not_final_and_auth_copy_removed(self):
        auth = self.fake_auth()
        original = auth.read_bytes()
        with patch.object(plans.shutil, "which", return_value="/usr/bin/codex"), \
                patch.object(plans.subprocess, "run", side_effect=self.live_runner()) as run:
            envelope = plans.generate_plan(public_input(), "C1", model="gpt-6-astra", path_count=3,
                live=True, output_path="plans/dev/C1.json", auth_path=auth)
        self.assertEqual(run.call_count, 6)
        self.assertEqual(envelope["metadata"]["provenance"], "codex_exec_development")
        self.assertEqual(envelope["metadata"]["accounting"]["source"], "captured_codex_exec_jsonl")
        self.assertIsNone(envelope["metadata"]["accounting"]["actual_model_requests"])
        self.assertEqual(auth.read_bytes(), original)
        self.assertEqual(plans.load_plan(self.root / "plans/dev/C1.json", "C1", 3), plain())
        self.assertFalse(list((self.root / "plans").rglob(".planner-*")))
        for path in (self.root / "plans").rglob("*.json"):
            self.assertNotIn("SECRET_", path.read_text())
            self.assertNotIn("PRIVATE_REASONING", path.read_text())

    def test_live_failure_and_timeout_keep_usage_without_secret_output(self):
        auth = self.fake_auth()
        for failure in ("timeout", "malformed", "exit"):
            with patch.object(plans.shutil, "which", return_value="/usr/bin/codex"), \
                    patch.object(plans.subprocess, "run", side_effect=self.live_runner(failure=failure)):
                with self.assertRaises(plans.PlanError):
                    plans.generate_plan(public_input(), "C1", model="gpt-6-astra", path_count=3,
                                        live=True, output_path=f"plans/{failure}/C1.json", auth_path=auth)
            self.assertFalse((self.root / f"plans/{failure}/C1.json").exists())
        results = [json.loads(p.read_text()) for p in (self.root / "plans/attempts").glob("*.json")
                   if not p.name.endswith(".started.json")]
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsNone(result["accounting"]["actual_model_requests"])
            self.assertEqual(result["accounting"]["token_usage"]["input_tokens"], 100)
            self.assertTrue(result["failure_detail"])
            self.assertNotIn("SECRET_", json.dumps(result))
            self.assertNotIn("PRIVATE_REASONING", json.dumps(result))

    def test_private_context_and_api_key_auth_fail_before_dispatch(self):
        auth = self.fake_auth()
        with patch.object(plans.shutil, "which", return_value="/usr/bin/codex"), \
                patch.object(plans.subprocess, "run", side_effect=self.live_runner(contaminated=True)) as run:
            with self.assertRaises(plans.PlanError):
                plans.generate_plan(public_input(), "C1", model="gpt-6-astra", path_count=3, live=True, auth_path=auth)
        self.assertEqual(run.call_count, 2)
        auth.write_text(json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": "SECRET_PAID_KEY"}))
        with patch.object(plans.shutil, "which", return_value="/usr/bin/codex"), \
                patch.object(plans.subprocess, "run", side_effect=self.live_runner()) as run:
            with self.assertRaises(plans.PlanError):
                plans.generate_plan(public_input(), "C1", model="gpt-6-astra", path_count=3, live=True, auth_path=auth)
        self.assertEqual(run.call_count, 2)
        for path in (self.root / "plans/attempts").glob("*.json"):
            audit = json.loads(path.read_text())
            self.assertEqual(audit["accounting"]["actual_model_requests"], 0)
            self.assertNotIn("SECRET_PAID_KEY", path.read_text())

    def test_import_bundle_cli_and_arena_tuple_api(self):
        public = self.root / "public.json"
        public.write_text(json.dumps(public_input()))
        paths = []
        with patch.object(plans.subprocess, "run") as run, redirect_stdout(io.StringIO()):
            for agent in plans.AGENT_IDS:
                raw = self.root / f"raw-{agent}.json"
                raw.write_text(json.dumps(plain(agent=agent)))
                out = self.root / "plans/dev" / f"{agent}.json"
                self.assertEqual(plans.main(["import", "--public-input", str(public), "--agent-id", agent,
                    "--model", "gpt-6-astra", "--path-count", "3", "--plan-file", str(raw), "--output", str(out)]), 0)
                paths.append(str(out))
            bundle = self.root / "plans/dev/bundle.json"
            self.assertEqual(plans.main(["bundle", "--public-input", str(public), "--path-count", "3",
                                        "--envelopes", *paths, "--output", str(bundle)]), 0)
        run.assert_not_called()
        loaded, provenance = plans.load_bundle(bundle, public_input(), 3)
        self.assertEqual(set(loaded), set(plans.AGENT_IDS))
        self.assertEqual(provenance["model"], "gpt-6-astra")
        self.assertIsNone(provenance["actual_model_requests"])
        self.assertTrue(all(v["provenance"] == "imported_development_plan" for v in provenance["agents"].values()))
        with self.assertRaises(plans.FinalBudgetError):
            plans.load_bundle(bundle, public_input(), 3, final=True)

    def test_bundle_rejects_wrong_agent_model_context_and_bare_claimed_provenance(self):
        envelopes = {agent: plans.make_envelope(plain(agent=agent), public_input(), model="gpt-6-astra", path_count=3)
                     for agent in plans.AGENT_IDS}
        for mutation in (
            lambda e: e.pop("A2"),
            lambda e: e.update(A2=e["A1"]),
            lambda e: e["A2"]["metadata"].update(model="other-model"),
            lambda e: e["A2"]["metadata"].update(provenance="codex-generated-trust-me"),
        ):
            changed = copy.deepcopy(envelopes)
            mutation(changed)
            with self.assertRaises(plans.PlanError):
                plans.save_bundle("plans/invalid.json", changed)
        bare = self.root / "bare.json"
        bare.write_text(json.dumps({"public_input_sha256": "claimed", "plans": {a: plain(agent=a) for a in plans.AGENT_IDS},
                                    "provenance": "Codex"}))
        with self.assertRaises(plans.PlanError):
            plans.load_bundle(bare, public_input(), 3)

    @unittest.skipUnless(os.environ.get("CODEX_PLANS_OFFLINE_PROBE") == "1", "optional localhost-only CLI wire probe")
    def test_local_cli_public_context_and_filesystem_boundary(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        bodies = []
        model = os.environ.get("CODEX_PLANS_PROBE_MODEL", "gpt-6-astra")

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"type":"response.failed","response":{"error":{"code":"probe_only","message":"Offline probe complete"}}}\n\n')

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            home, work = self.root / "home", self.root / "work"
            home.mkdir()
            (home / ".codex").mkdir()
            work.mkdir()
            (work / ".git").mkdir()
            instructions = self.root / "instructions.txt"
            instructions.write_text("Only produce public route plans.")
            env = plans._private_environment(home, self.root)
            config = plans._cli_config(instructions)
            executable = plans.shutil.which("codex")
            probe = subprocess.run([executable, "debug", "prompt-input", *config, "-c", "model=" + json.dumps(model), "PROBE"],
                                   cwd=work, env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(probe.returncode, 0, probe.stderr)
            messages = json.loads(probe.stdout)
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["content"], [{"type": "input_text", "text": "PROBE"}])
            provider = ('model_providers.probe={name="probe",base_url="http://127.0.0.1:%s/v1",'
                        'wire_api="responses",requires_openai_auth=false,request_max_retries=0,stream_max_retries=0}') % server.server_port
            result = subprocess.run([executable, "exec", "--ignore-user-config", "--ignore-rules", "--strict-config",
                                     "--ephemeral", "--skip-git-repo-check", "--json", "--model", model,
                                     *config, "-c", 'model_provider="probe"', "-c", provider, "PROBE"],
                                    cwd=work, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=20)
            self.assertEqual(len(bodies), 1, result.stderr)
            # Native profiles can carry instructions as a developer message and
            # retain apply_patch. Filesystem protection is verified independently.
            body = bodies[0]
            messages = [item for item in body["input"] if item.get("type") == "message"]
            user = [m for m in messages if m["role"] == "user"]
            self.assertEqual(len(user), 1)
            self.assertEqual(user[0]["content"], [{"type": "input_text", "text": "PROBE"}])
            developer = [m for m in messages if m["role"] == "developer"]
            if developer:
                self.assertEqual(len(messages), 2)
                self.assertEqual(developer[0]["content"], [{"type": "input_text", "text": instructions.read_text()}])
            else:
                self.assertEqual(len(messages), 1)
                self.assertEqual(body["instructions"], instructions.read_text())
            plans._verify_filesystem_boundary(executable, config, env, work, self.root)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main(verbosity=2)
