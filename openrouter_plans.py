"""OpenRouter-backed saved route-plan generation for CoopMEV-Lite.

The online arena still uses frozen PPO for candidate routes/amounts and deterministic
execution code. OpenRouter is used only before timed episodes to generate saved route-order
preferences from the approved public planning input. No evaluation outcome is sent.
Credentials are read only from OPENROUTER_API_KEY and are never serialized.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.request

import codex_plans as core
from openrouter_config import (OPENROUTER_API_KEY_ENV, OPENROUTER_API_URL, OPENROUTER_MODEL,
                               OPENROUTER_MAX_COMPLETION_TOKENS, OPENROUTER_REASONING_EFFORT,
                               OPENROUTER_SITE_URL, OPENROUTER_APP_NAME, get_api_key)

# Preserve the arena-facing API.
PlanError = core.PlanError
AGENT_IDS = core.AGENT_IDS
PATH_COUNT = core.PATH_COUNT
FINAL_REQUEST_BUDGET = core.FINAL_REQUEST_BUDGET
validate_plan = core.validate_plan
rank_candidates = core.rank_candidates
prepare_request = core.prepare_request
load_plan = core.load_plan
load_bundle = core.load_bundle
save_plan = core.save_plan
save_bundle = core.save_bundle
validate_public_input = core.validate_public_input
_load_json = core._load_json


def _extract_json_object(text: str) -> dict[str, Any]:
    """Accept a JSON object, with a conservative fallback for fenced JSON."""
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        if len(lines) >= 3:
            value = "\n".join(lines[1:-1]).strip()
            if value.lower().startswith("json\n"):
                value = value[5:].lstrip()
    parsed = _load_json(value)
    if type(parsed) is not dict:
        raise PlanError("OpenRouter response must be one JSON object")
    return parsed


def _usage_audit(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage") if type(payload) is dict else None
    token_usage = None
    if type(usage) is dict:
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if type(prompt) is int and prompt >= 0 and type(completion) is int and completion >= 0:
            token_usage = {"input_tokens": prompt, "cached_input_tokens": 0, "output_tokens": completion}
    # Reuse the existing envelope schema. One HTTP POST is one application-level request;
    # the legacy envelope intentionally leaves provider-internal request accounting unverified.
    return {
        "cli_invocations": None,
        "completed_turns": 1,
        "token_usage": token_usage,
        "token_usage_status": "reported" if token_usage else "unknown",
        "failures": [],
        "actual_model_requests": None,
        "request_count_status": "unverifiable",
        "source": "not_supplied",
        "request_count_reason": core.ACCOUNTING_LIMITATION,
    }


def generate_plan(public_input: dict[str, Any], agent_id: str, *, model: str = OPENROUTER_MODEL,
                  path_count: int = PATH_COUNT, live: bool = False,
                  output_path: str | Path | None = None, timeout: float = 180) -> dict[str, Any]:
    """Generate one development plan through OpenRouter Chat Completions.

    Generation is opt-in via live=True. There are no automatic retries, which keeps
    application-level request accounting simple. The API key is environment-only.
    """
    if not live:
        raise PlanError("OpenRouter dispatch is disabled unless --live is supplied")
    if model != OPENROUTER_MODEL:
        raise PlanError(f"this run is pinned to configured OPENROUTER_MODEL={OPENROUTER_MODEL!r}; got {model!r}")
    request = prepare_request(public_input, agent_id, path_count)
    target = core._plans_path(output_path or f"plans/openrouter-{int(time.time())}/{agent_id}.json")
    if target.exists():
        raise FileExistsError("refusing to overwrite an existing saved plan")
    key = get_api_key()
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only the requested JSON route-allocation plan. Do not use tools."},
            {"role": "user", "content": request["prompt"]},
        ],
        "temperature": 0,
        "max_completion_tokens": OPENROUTER_MAX_COMPLETION_TOKENS,
        "reasoning": {"effort": OPENROUTER_REASONING_EFFORT, "exclude": True},
        # Enforce the exact CoopMEV plan shape at generation time. JSON Object Mode
        # only guarantees syntactically valid JSON; it does not prevent the model
        # from inventing extra fields, which the local research validator rejects.
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "coopmev_route_plan",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "schema_version": {
                            "type": "string",
                            "enum": [core.PLAN_SCHEMA_VERSION],
                        },
                        "agent_id": {
                            "type": "string",
                            "enum": [agent_id],
                        },
                        "route_preferences": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "schema_version", "agent_id",
                        "route_preferences", "reason",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        # Structured-output support is endpoint-specific on OpenRouter. Require it
        # rather than silently routing to an endpoint that ignores response_format.
        "provider": {"require_parameters": True},
    }
    http_request = urllib.request.Request(
        OPENROUTER_API_URL,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Some API gateways reject Python urllib's default User-Agent even when
            # the same authenticated request succeeds with curl.
            "User-Agent": "CoopMEV-Lite/1.0 (+OpenRouter Chat Completions)",
            **({"HTTP-Referer": OPENROUTER_SITE_URL} if OPENROUTER_SITE_URL else {}),
            **({"X-Title": OPENROUTER_APP_NAME} if OPENROUTER_APP_NAME else {}),
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            payload = _load_json(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Read diagnostics only in memory, sanitize/truncate, and never persist them.
        # Provider error bodies do not contain our Authorization header, but avoid dumping
        # arbitrary provider payloads into research artifacts.
        detail = ""
        try:
            raw = exc.read().decode("utf-8", errors="replace")
            parsed_error = json.loads(raw)
            if isinstance(parsed_error, dict):
                err = parsed_error.get("error")
                if isinstance(err, dict):
                    msg = err.get("message")
                    typ = err.get("type")
                    code = err.get("code")
                    parts = [str(x) for x in (typ, code, msg) if x]
                    detail = ": ".join(parts)[:600]
        except Exception:
            detail = ""
        suffix = f" ({detail})" if detail else ""
        raise PlanError(
            f"OpenRouter HTTP error {exc.code}{suffix}; credential and response body were not saved"
        ) from None
    except urllib.error.URLError as exc:
        raise PlanError(f"OpenRouter connection failed: {exc.reason}") from None
    finally:
        # Drop the local reference as soon as the request has been formed/sent.
        key = ""

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise PlanError("OpenRouter response did not contain choices[0].message.content") from None
    plain = validate_plan(_extract_json_object(content), agent_id, path_count)
    envelope = core.make_envelope(
        plain, public_input, model=model, path_count=path_count,
        audit=_usage_audit(payload), generation_seconds=time.monotonic() - started,
    )
    # Existing envelope validator accepts imported development provenance. This truthfully
    # records the requested model while avoiding a false claim that Codex CLI generated it.
    envelope["metadata"]["transport"] = "openrouter_chat_completions"  # removed before legacy save validation
    envelope["metadata"].pop("transport")
    saved = save_plan(target, envelope)
    return envelope


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="generate", choices=("generate", "bundle"))
    parser.add_argument("--public-input", type=Path, required=True)
    parser.add_argument("--agent-id", choices=AGENT_IDS)
    parser.add_argument("--model", default=OPENROUTER_MODEL)
    parser.add_argument("--path-count", type=int, default=PATH_COUNT)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envelopes", type=Path, nargs=4)
    args = parser.parse_args(argv)
    try:
        public = _load_json(args.public_input.read_text(encoding="utf-8"))
        if args.action == "generate":
            if args.agent_id is None:
                raise PlanError("--agent-id is required for generate")
            envelope = generate_plan(public, args.agent_id, model=args.model, path_count=args.path_count,
                                     live=args.live, output_path=args.output)
            print(f"Saved OpenRouter development plan for {envelope['plan']['agent_id']}: {args.output}")
        else:
            if args.envelopes is None or args.live:
                raise PlanError("bundle requires exactly four --envelopes and no --live")
            envelopes = {}
            for path in args.envelopes:
                envelope = _load_json(path.read_text(encoding="utf-8"))
                agent = envelope.get("plan", {}).get("agent_id") if type(envelope) is dict else None
                core._validate_envelope(envelope, agent, args.path_count, public)
                if agent in envelopes:
                    raise PlanError("duplicate bundle agent")
                envelopes[agent] = envelope
            print(f"Saved OpenRouter development bundle: {save_bundle(args.output, envelopes)}")
    except (PlanError, RuntimeError, OSError) as exc:
        print(f"Plan operation refused: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
