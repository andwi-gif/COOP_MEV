# CoopMEV-Lite

CoopMEV-Lite is the COSC 723 development prototype for evaluating communication between two cooperative PPO agents (C1/C2) against two independent competing searchers (A1/A2) in a simulated Uniswap V2/SushiSwap V2-style CPMM market. The current development branch replaces the original Codex planner transport with OpenRouter using the pinned model `openai/gpt-oss-120b`; the PPO checkpoint, shared-arena execution workflow, communication conditions, deterministic execution checker, and saved-plan-before-evaluation design are preserved.

> **Development status:** the OpenRouter dry-run path is operational. This repository is not yet the final 23 settings x 10 seeds evaluation. The OpenRouter/model substitution must be disclosed and confirmed before the final experimental protocol is frozen.

## Development dry-run result

A seed-30 development run completed all five conditions with the saved OpenRouter plan bundle. In that single development run, `Proposed` produced the largest observed team profit among the five conditions. This is a smoke/dry-run observation only; it is **not** the final paired result and does not establish the research hypothesis. Final evidence requires the approved 23-setting x 10-seed design, paired ON-minus-OFF analysis by sweep, and bootstrap intervals over complete seed groups.

## System workflow

```text
openrouter_config.py
    │  provider/model/runtime configuration; API key comes only from environment
    ▼
openrouter_plans.py
    │  BEFORE evaluation: send public planning input to OpenRouter
    │  validate structured route-preference plan locally
    ▼
plans/<setting>/{C1,C2,A1,A2}.json
    │
    ▼
openrouter_plans.py bundle
    │  verify model/input/prompt/schema/agent consistency
    ▼
plans/<setting>/bundle.json
    │
    │  ──────────────── TIMED EPISODE ────────────────
    ▼
coopmev_arena.py
    │
    ├── ppo_runtime.py
    │      ├── ppo_250000_model.zip        (frozen PPO checkpoint)
    │      └── DATASET/*                   (frozen simulator inputs)
    │
    ├── PPO candidate routes + route-specific amounts from current state
    │
    ├── saved OpenRouter LLM plan -> candidate priority
    │
    ├── C1 -> C2 route message -> duplicate-route avoidance when enabled
    │
    ├── deterministic readiness/tie-breaking queue
    │
    ▼
execution_checker.py
    │  independently checks swaps, fees, gas, wallets/reserves and atomic rejection
    ▼
atomic commit/reject + machine-readable run evidence
    │
    ▼
runs/*.json

ppr01_ablation_driver_v2.09.py
    └── retained PPO-only/RPE driver; not the shared-arena orchestration above
```

OpenRouter is **not called inside the timed episode**. This preserves the approved saved-plan workflow: model planning occurs before evaluation; PPO supplies online route candidates and amounts; deterministic code handles communication, queueing, checking, and execution.

## Important files

| File / directory | Purpose | Needed for development rerun? |
|---|---|---|
| `README.md` | Repository entry point, workflow, checklist, run instructions | Yes |
| `openrouter_config.py` | OpenRouter endpoint/model/token/reasoning configuration | Yes |
| `openrouter_plans.py` | Generate, validate, save, and bundle OpenRouter route plans | Yes |
| `coopmev_arena.py` | Four-agent shared-arena orchestration and B1/B2/B3/Proposed/no-team-plan conditions | Yes |
| `ppo_runtime.py` | Loads frozen PPO and produces route candidates/amounts | Yes |
| `execution_checker.py` | Deterministic independent trade/outcome checker | Yes |
| `ppo_250000_model.zip` | Frozen PPO checkpoint | Yes |
| `DATASET/` | Pool, token-price, wallet and related simulator inputs | Yes |
| `tests/` | Regression, planner, arena, PPO-runtime, and checker tests | Yes |
| `plans/` | Public planning inputs, checked plans, bundles, and development attempts | For reproducibility |
| `runs/` | Development evidence and machine-readable outputs | For reproducibility |
| `ppr01_ablation_driver_v2.09.py` | Reused PPO-only/RPE driver retained for provenance | Context/reuse |
| `codex_plans.py` | Legacy Codex planner retained for development history/reference | No for OpenRouter run |
| `PROJECT_REQUIREMENTS_AND_RUN.md` | Detailed protocol/runbook and final-freeze notes | Yes |
| `README_STEP56.md` | Historical Step-5/6 development note; legacy Codex-era details | Historical only |

## Professor-request checklist

| Requirement from approved contract / email | Development implementation/evidence | Status before final evaluation |
|---|---|---|
| Bounded claim: fixed cooperative pair vs two competing searchers | Conditions and threat-model wording retained | Preserve |
| B2 is primary communication-OFF control; B1 is context | Implemented in `coopmev_arena.py` | Preserve |
| Separate communication value from LLM-plan value | `Proposed` vs B2; `Proposed` vs `PPO-sharing-no-team-plan` | Preserve |
| Same PPO/checkpoint, starting state, rivals and execution rules in matched comparisons | Shared arena and frozen PPO path | Verify/freeze hashes |
| Offline planner cost/latency separate from online timing | Plan generation is before timed episode | Record final request/token/latency ledger |
| Explicit C6 boundary | Documented in `PROJECT_REQUIREMENTS_AND_RUN.md` | Freeze before final runs |
| Independent deterministic execution checker | `execution_checker.py` plus known-answer tests | Preserve and report |
| Exact 23-setting design and defaults | Documented in protocol runbook | Final run still pending |
| Seeds 40-49, paired by complete seed groups | Final protocol documented | Final run still pending |
| Report repeated reference settings by sweep | Final protocol documented | Final analysis pending |
| Distinguish invalid proposals, rejected trades, invalid committed trades | Arena/checker metrics retain distinct fields | Preserve |
| Freeze state, PPO, prompts, checked plans, model, rules and evaluation settings | Freeze procedure documented | Pending final freeze |
| Record model/code/state hashes, prompts, seeds, outcomes and costs | Development artifacts contain provenance fields | Complete final ledger/hashes |
| Preserve development history, failed attempts and AI-use record | `plans/attempts/`, `runs/`, Git history/AI-use record | Continue preserving |
| Exclude credentials/secrets | Environment-only API key; `.gitignore` excludes `.env`; secret scan performed | Re-scan before push |
| Complete running system in GitHub repository accessible to professor | This repository is structured for development push | Development push ready; final results pending |
| Five-minute live/recorded demo | Workflow supports reproducible dry run | Demo pending |
| Planner change from approved Codex wording | OpenRouter `openai/gpt-oss-120b` is explicitly disclosed | Obtain/document professor confirmation before final freeze |

## Environment and installation

The frozen PPO checkpoint embeds the environment metadata used when it was saved: **Python 3.10.20, Stable-Baselines3 2.9.0, PyTorch 2.12.1+cu130, NumPy 2.2.6, Cloudpickle 3.1.2, and Gymnasium 1.3.0**. The repository now includes `requirements.txt` and `environment.yml`. PyTorch is intentionally installed separately because CUDA builds are platform/HPC specific; CPU-only or a site-supported CUDA build can be used for inference.

### Option A — Conda (recommended)

```bash
git clone https://github.com/andwi-gif/C723_COOPMEV.git
cd C723_COOPMEV
conda env create -f environment.yml
conda activate coopmev-lite

# Install PyTorch using the build appropriate for the machine/HPC.
# Example generic pip installation; use the HPC-supported CUDA command if required.
python -m pip install torch

# Verify the important runtime imports.
python - <<'PY'
import numpy, torch, gymnasium, stable_baselines3, pandas, openpyxl
print("numpy", numpy.__version__)
print("torch", torch.__version__)
print("gymnasium", gymnasium.__version__)
print("stable_baselines3", stable_baselines3.__version__)
PY
```

### Option B — existing HPC Conda environment

If the project environment already loads the frozen PPO successfully, keep it and install only missing packages:

```bash
python --version
python -m pip install -r requirements.txt
# Install/retain the site-supported PyTorch build separately.
```

`requests` and `web3` are included because the committed `DATASET/` contains dataset-regeneration utilities that import them. They are not needed to rerun the frozen dataset itself. No external OpenRouter Python SDK is required: the planner uses Python's standard-library HTTP client.

### Expected repository layout

Run commands from the repository root (the directory containing `coopmev_arena.py`):

```text
C723_COOPMEV/
├── README.md
├── requirements.txt
├── environment.yml
├── .env.example
├── openrouter_config.py
├── openrouter_plans.py
├── coopmev_arena.py
├── ppo_runtime.py
├── execution_checker.py
├── ppo_250000_model.zip
├── DATASET/
├── tests/
├── plans/
└── runs/
```

Do not move `ppo_250000_model.zip` or `DATASET/` unless the corresponding code paths are deliberately updated. Generated plan/run directories are created or populated by the commands below.

## Quick development rerun

After preparing the environment above, configure OpenRouter. Do not place the API key in source code or a committed `.env` file.

```bash
export OPENROUTER_API_KEY='YOUR_OWN_OPENROUTER_KEY'
export OPENROUTER_MODEL='openai/gpt-oss-120b'
export OPENROUTER_MAX_COMPLETION_TOKENS='512'
export OPENROUTER_REASONING_EFFORT='low'
```

Run local checks:

```bash
export PYTHONDONTWRITEBYTECODE=1
python -B tests/test_execution_checker.py
python -B tests/test_openrouter_plans.py
python -B tests/test_coopmev_arena.py
python -B coopmev_arena.py --dry-run
```

Export one public development planning input:

```bash
python -B coopmev_arena.py --export-planning-input plans/public_openrouter_example.json
```

Generate four checked saved plans:

```bash
python -B openrouter_plans.py generate --public-input plans/public_openrouter_example.json --agent-id C1 --live --output plans/openrouter_example/C1.json
python -B openrouter_plans.py generate --public-input plans/public_openrouter_example.json --agent-id C2 --live --output plans/openrouter_example/C2.json
python -B openrouter_plans.py generate --public-input plans/public_openrouter_example.json --agent-id A1 --live --output plans/openrouter_example/A1.json
python -B openrouter_plans.py generate --public-input plans/public_openrouter_example.json --agent-id A2 --live --output plans/openrouter_example/A2.json
```

Bundle and run the shared arena:

```bash
python -B openrouter_plans.py bundle \
  --public-input plans/public_openrouter_example.json \
  --envelopes plans/openrouter_example/C1.json plans/openrouter_example/C2.json plans/openrouter_example/A1.json plans/openrouter_example/A2.json \
  --output plans/openrouter_example/bundle.json

python -B coopmev_arena.py --plans plans/openrouter_example/bundle.json --condition all --seed 30
```

## Before the final experiment

Do not interpret the development seed-30 dry run as final evidence. Before the final evaluation, obtain/document approval for the OpenRouter/model substitution, freeze all protocol artifacts, implement/verify the final model-request ledger (maximum 92 requests including retries), use the agreed seeds 40-49 across all 23 settings, keep development and evaluation artifacts separate, and perform the paired per-sweep analysis required by the contract.

## Security and repository hygiene

No real provider key should be committed. `.env.example` contains placeholders only, while `.gitignore` excludes `.env`, `.env.*`, `__pycache__/`, and `*.pyc`. Before every GitHub push, inspect `git diff --cached`, run a secret scanner if available, and verify that shell history, editor settings, SSH material, browser profiles, provider credentials, and private course/email documents are not staged.

For the detailed protocol mapping and C6 boundary, see [`PROJECT_REQUIREMENTS_AND_RUN.md`](PROJECT_REQUIREMENTS_AND_RUN.md).
