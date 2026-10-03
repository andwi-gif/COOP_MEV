# CoopMEV-Lite — implementation checklist and OpenRouter runbook

## Important protocol change

This code revision replaces **Codex CLI planning** with the OpenRouter Chat Completions API using the pinned model ID **`openai/gpt-oss-120b`**. The approved contract says that integrated agents use a fixed Codex model ID and the professor's approval explicitly refers to the Codex-plan comparison. Therefore, this implementation change must be disclosed before treating the final experimental protocol as frozen. Do not relabel old Codex plans as OpenRouter plans, and do not mix old and new planner artifacts in a matched comparison.

## Professor requirements implemented / to preserve

1. **Bound the claim.** The experiment tests the value of communication between fixed teammates C1/C2 against two independent competing searchers A1/A2. It does not claim general MEV protection, sandwich/front-running prevention, online LLM adaptation, or live-market profitability.
2. **Separate communication from planner contribution.** B2 is the primary communication-OFF control. B1 is context only. `Proposed` versus B2 measures communication value; `Proposed` versus `PPO-sharing-no-team-plan` measures saved-plan value while preserving PPO, rivals, starting state, collision avoidance, and execution rules.
3. **Match resources and timing.** Planner generation happens before timed episodes. Record model ID, calls/requests, token usage when returned, failures, and generation latency separately from online decision/execution time. Do not describe a saved plan as an online LLM decision.
4. **Freeze experiment and keep checker independent.** Freeze state snapshots, PPO checkpoint, prompts, checked plans, candidate-selection rules, timing rules, seeds, and evaluation settings before final runs. Keep development cases separate from evaluation cases. Validate the deterministic execution checker with independently calculated known-answer cases.
5. **Preserve the 23-setting design and pairing.** 10 wallet settings + 5 C6 settings + 4 gas settings + 4 liquidity settings = 23 settings. Use seeds 40–49 for the agreed final design, retain repeated reference settings within their sweeps, report paired ON-minus-OFF differences by sweep, and bootstrap complete seed groups rather than treating 230 rows as independent experiments. Keep invalid proposals, rejected trades, and invalid committed trades distinct.
6. **Make new COSC 723 work and reproducibility explicit.** Reused: simulator, route data, selected PPO checkpoint, scenario definitions. New: multi-agent adapter, route messages, saved-plan interface, no-planner control, independent checker tests, paired experiments, and demo. Preserve code/state hashes, model version, prompt/schema, seeds, raw outcomes, costs, dependencies, tests, configurations, AI-use record, failed attempts, and development history. Exclude credentials/secrets.

The professor's final implementation note also requires the **C6 timing boundary** to be explicit, seed-group pairing and per-sweep reporting (including repeated references) to be retained, and the complete running system to be submitted in a GitHub repository with development commit history and AI-use records.

## C6 timing boundary used by the current arena

C6 is one active-time budget for the whole shared arena, not four independent per-agent budgets. Observation, global opportunity checks, PPO inference, candidate selection, team-message handling, queue work, execution checking, commit, and online event recording count toward C6. Model-plan generation, loading/warm-up, optional post-hoc rival-attribution diagnostics, and final result serialization are outside C6 and must be reported separately.

## API key configuration

The code **does not contain an API key**. Configuration is in `openrouter_config.py`, and the credential is read only from the environment variable `OPENROUTER_API_KEY`. `.env.example` contains placeholders only. A professor/reviewer must use their own OpenRouter key.

```bash
export OPENROUTER_API_KEY='YOUR_OWN_OPENROUTER_KEY'
export OPENROUTER_MODEL='openai/gpt-oss-120b'
export OPENROUTER_MAX_COMPLETION_TOKENS='512'
export OPENROUTER_REASONING_EFFORT='low'
```

Never commit `.env`, shell history containing a key, provider credentials, or generated logs containing secrets. If a key has ever been pasted into chat, email, a repository, or another shared location, revoke/rotate it in the provider console and use the replacement only through the environment.

## Model and API

Pinned model: `openai/gpt-oss-120b`. The planner calls OpenRouter's OpenAI-compatible Chat Completions endpoint directly using Python's standard library, requests JSON-object output, uses `temperature=0`, caps completion at 512 tokens by default, uses `reasoning_effort=low`, performs no automatic retry, validates the returned plan locally, and saves only the checked plan envelope. PPO still supplies route candidates and route-specific amounts; deterministic code performs message logic, queueing, checking, and execution.

## Environment setup and repository layout

The frozen PPO archive contains its original save-environment metadata: Python 3.10.20, Stable-Baselines3 2.9.0, PyTorch 2.12.1+cu130, NumPy 2.2.6, Cloudpickle 3.1.2, and Gymnasium 1.3.0. `requirements.txt` pins the non-PyTorch checkpoint-critical packages and declares spreadsheet/runtime utilities; `environment.yml` creates the reproducible Python/Conda base. PyTorch is installed separately so the reviewer can select the build appropriate for the target CPU/CUDA/HPC platform.

Recommended fresh setup:

```bash
git clone https://github.com/andwi-gif/C723_COOPMEV.git
cd C723_COOPMEV
conda env create -f environment.yml
conda activate coopmev-lite
python -m pip install torch   # replace with the institution/HPC CUDA-specific command when needed
```

The commands in this document assume the working directory is the repository root containing `coopmev_arena.py`, with `ppo_250000_model.zip` and `DATASET/` at their committed relative paths. The `plans/` and `runs/` directories hold planning/evaluation artifacts. `requests` and `web3` are needed only by dataset-regeneration utilities; frozen-dataset evaluation does not contact those data sources.

## Running from the repository root

First run the local tests and dry-run (these do not require an OpenRouter key):

```bash
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -B tests/test_execution_checker.py
python -B tests/test_openrouter_plans.py
python -B tests/test_coopmev_arena.py
python -B coopmev_arena.py --dry-run
python -B coopmev_arena.py --fixture-plans --condition all --seed 30
```

Export the public planning input for a development setting:

```bash
python -B coopmev_arena.py --export-planning-input plans/public_openrouter_example.json
```

Generate one saved OpenRouter plan per agent (each command makes one application-level API request and requires `--live`):

```bash
python -B openrouter_plans.py generate --public-input plans/public_openrouter_example.json --agent-id C1 --live --output plans/openrouter_example/C1.json
python -B openrouter_plans.py generate --public-input plans/public_openrouter_example.json --agent-id C2 --live --output plans/openrouter_example/C2.json
python -B openrouter_plans.py generate --public-input plans/public_openrouter_example.json --agent-id A1 --live --output plans/openrouter_example/A1.json
python -B openrouter_plans.py generate --public-input plans/public_openrouter_example.json --agent-id A2 --live --output plans/openrouter_example/A2.json
```

Bundle the four checked envelopes:

```bash
python -B openrouter_plans.py bundle --public-input plans/public_openrouter_example.json \
  --envelopes plans/openrouter_example/C1.json plans/openrouter_example/C2.json plans/openrouter_example/A1.json plans/openrouter_example/A2.json \
  --output plans/openrouter_example/bundle.json
```

Run the shared arena using the saved bundle. This step makes **no model API call**:

```bash
python -B coopmev_arena.py --plans plans/openrouter_example/bundle.json --condition all --seed 30
```

## Before final evaluation

Do not regenerate plans after inspecting evaluation profit. Do not use development seeds as final evidence. Freeze the model ID, public state snapshots, prompts, checked plans, PPO checkpoint, code, timing rules, candidate-selection rules, and evaluation settings first. The approved contract caps final planning at 92 model requests across 4 agents × 23 settings including retries; because this OpenRouter migration changes the planner transport/model from the approved Codex wording, obtain/document professor confirmation and implement a final request ledger before declaring final eligibility.

## OpenRouter reproducibility note

OpenRouter can route the same model slug across multiple upstream providers. For development this implementation uses `provider.require_parameters=true` so the selected endpoint must support the requested parameters. Before the final frozen experiment, record the upstream provider returned by OpenRouter if available and, if the protocol requires exact provider reproducibility, pin an approved provider consistently across all four agents and all 23 settings rather than changing routing mid-experiment.
