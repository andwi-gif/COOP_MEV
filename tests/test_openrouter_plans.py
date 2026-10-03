import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import openrouter_plans as gp


def public_input():
    pools = [
        {"pool_id":"p0","token0":"A","token1":"B","reserve0":1000.0,"reserve1":1000.0,"fee":0.003},
        {"pool_id":"p1","token0":"B","token1":"C","reserve0":1000.0,"reserve1":1000.0,"fee":0.003},
        {"pool_id":"p2","token0":"C","token1":"A","reserve0":1000.0,"reserve1":1000.0,"fee":0.003},
    ]
    routes = [{"route_index":0,"tokens":["A","B","C","A"],"pool_ids":["p0","p1","p2"]}]
    scenario = {"dimension":"test","ablation_value":1.0,"ablation_label":"test","balance_usd":2000.0,
                "c3_liquidity_depth":1.0,"gas_multiplier":1.0,"c6_latency_budget_seconds":12.0}
    return {"schema_version":"coopmev-public-v1","scenario":scenario,"routes":routes,"pools":pools}


class FakeResponse:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return json.dumps(self.payload).encode()


class OpenRouterTests(unittest.TestCase):
    def test_missing_key_is_refused(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                gp.generate_plan(public_input(), "C1", path_count=1, live=True, output_path="plans/test-missing/C1.json")

    def test_mocked_generation_never_serializes_key(self):
        payload = {"choices":[{"message":{"content":json.dumps({"schema_version":"coopmev-plan-v1","agent_id":"C1","route_preferences":[0],"reason":"Prefer route 0."})}}],
                   "usage":{"prompt_tokens":12,"completion_tokens":9}}
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {"OPENROUTER_API_KEY":"sk-or-v1-TEST_SECRET"}, clear=False):
            old = gp.core.HERE
            gp.core.HERE = Path(td)
            try:
                with mock.patch("urllib.request.urlopen", return_value=FakeResponse(payload)) as mocked_urlopen:
                    env = gp.generate_plan(public_input(), "C1", path_count=1, live=True, output_path="plans/test/C1.json")
                text = (Path(td)/"plans/test/C1.json").read_text()
                self.assertNotIn("sk-or-v1-TEST_SECRET", text)
                self.assertEqual(env["metadata"]["model"], "openai/gpt-oss-120b")
                self.assertEqual(env["plan"]["route_preferences"], [0])
                req = mocked_urlopen.call_args.args[0]
                body = json.loads(req.data.decode())
                self.assertEqual(req.full_url, "https://openrouter.ai/api/v1/chat/completions")
                self.assertEqual(body["model"], "openai/gpt-oss-120b")
                self.assertEqual(body["reasoning"], {"effort": "low", "exclude": True})
                self.assertEqual(body["provider"], {"require_parameters": True})
                rf = body["response_format"]
                self.assertEqual(rf["type"], "json_schema")
                self.assertTrue(rf["json_schema"]["strict"])
                schema = rf["json_schema"]["schema"]
                self.assertEqual(set(schema["required"]), {"schema_version", "agent_id", "route_preferences", "reason"})
                self.assertFalse(schema["additionalProperties"])
                self.assertEqual(schema["properties"]["agent_id"]["enum"], ["C1"])
                self.assertNotIn("sk-or-v1-TEST_SECRET", req.data.decode())
            finally:
                gp.core.HERE = old

if __name__ == "__main__": unittest.main(verbosity=2)
