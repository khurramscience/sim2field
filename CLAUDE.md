# CLAUDE.md — Sim2Field

> This file is auto-loaded by Claude Code as project context. It captures the full
> vision, every requirement gathered so far, what is already built, and the exact
> next tasks. Read it before acting.

## What Sim2Field is

**Scan the field. See it fail. Fix it before it ships.**

Sim2Field turns a single scan of a real robot deployment site into a diverse,
adversarial, physics-grounded evaluation suite, runs candidate robot policies across
every generated variation, and returns a deployment report that predicts **where, how,
and with what confidence each policy will fail on that specific floor** — before the
robot touches hardware.

Built for the Google DeepMind Enterprise Build Day 2026 hackathon.
Repo: https://github.com/khurramscience/sim2field

## Core stance (do not drift from this)

The product is **not** a perfect simulator — sim-to-real friction is unsolved, and we
must not claim otherwise. Sim2Field is the **failure-discovery and ranking layer**. Its
value is measured by three things, never by visual realism:
1. **Discrimination** — does it separate good policies from bad?
2. **Sim→real transfer** — does its ranking match reality? (Spearman ρ vs BusyBox real episodes.)
3. **Calibration** — does model uncertainty predict failure? (AUROC.)

Honesty constraints to preserve in all copy and demos:
- Don't claim solved sim2real; we *sweep* unknown physics (e.g. friction) rather than guess one value.
- Genie 3 is navigation/scene generation, **not** a physics engine. MuJoCo owns contact dynamics.
- Current leaderboard/heatmap numbers come from the **mock executor**; they are realistic
  placeholders until real policies are run (Phase B).

## Original asks / requirements (from the project owner)

- RL evals / policy evals **before deployment, after training**.
- **Combine existing sims** and test which physics is better; move fast; scenery planning + case generation.
- Lean on **Google/DeepMind stack**: Gemini, Veo, **Genie 3**, MuJoCo, plus Street View
  scenery (since 2007), World Labs, and **imagine.io SimReady-Kitchens** (a collaborator's
  structured-world layer: https://huggingface.co/datasets/imagineio/PhysicalAI-SimReady-Kitchens-v1).
- Analyze **Genesis** (github.com/Genesis-Embodied-AI/genesis-world) for useful features. *(TODO — see NEXT_STEPS.)*
- Sim tradeoffs to respect: **MuJoCo** = best physics / slow on GPU; **Isaac** = great GPU / weak physics;
  **Newton** = promising middle ground.
- Benchmark inspiration: **Microsoft BusyBox** (affordance generalization, reconfigurable, 1000+ trajectories, HDF5),
  **DINO-WM** (world model, zero-shot MPC planning), **MolmoAct 2** (open ARM; SimplerEnv/LIBERO).
- Arena UX in the spirit of **LLM arenas / Pavlov's List** (pavlovslist.com).
- **Mobile web app**: scan/record the field → get RL environment variations → report with
  snapshots + analytics → tap a failure to inspect *why* and *how* it failed.
- Plan **as a scientist**: clear steps, benchmarks, metrics.
- Build in two phases (below).

## The six perturbation axes (the IP)

The Gemini Scenario Planner authors a grid across these, biased toward HARD cases:
1. **Spatial** — reposition affordances/objects (BusyBox-style).
2. **Physics** — sweep friction, mass, restitution.
3. **Dynamics** — slip, drop, external push (catches slow-replanning policies like batch-action MolmoAct 2).
4. **Perception** — lighting, texture, camera pose, sensor noise, partial observability.
5. **Language** — paraphrase the instruction.
6. **Distractor** — clutter / novel objects (Google Scanned Objects).

## Architecture

```
Scan → Scenario Planner (Gemini) → World Builder (Genie3+Veo+ScannedObjects) →
Multi-Sim Executor (MuJoCo truth + Genesis cross-check) → Scorer → Arena + Report (mobile)
```

Repo layout:
```
planner/generate_scenarios.py   Gemini planner; deterministic fallback if no GEMINI_API_KEY
worldbuilder/build_scene.py     STUB — Genie3/Veo/ScannedObjects → sim-ready variants
executor/run_rollouts.py        mock rollouts now; Phase-B seam for real MuJoCo/policy loop
scorer/build_report.py          success, heatmap, sim→real Spearman, calibration AUROC (pure python)
app/index.html                  mobile prototype (scan→envs→arena→report→inspect)
schemas/scenario.schema.json    scenario grid + rollout record contract
data/                           kitchen.json scene, grid.json, rollouts/, report.json, busybox_truth.json
```

## How to run (Phase A — works with no API key, no GPU)

```bash
pip install -r requirements.txt
python planner/generate_scenarios.py --scene data/kitchen.json --out data/grid.json --n 24
for p in molmoact2 dino-wm pi05 scripted; do python executor/run_rollouts.py --grid data/grid.json --policy $p --out data/rollouts; done
python scorer/build_report.py --rollouts data/rollouts --truth data/busybox_truth.json --out data/report.json
```

## Status

- [x] Mobile prototype (clickable, 6 screens incl. tap-to-inspect failure).
- [x] Scenario Planner (Gemini + fallback), schema, mock executor, scorer with real metrics.
- [x] End-to-end Phase A pipeline produces grid + rollouts + report.
- [ ] Pushed to GitHub + Pages enabled.
- [ ] App reads real data/report.json instead of hardcoded numbers.
- [ ] Phase B: real policy in MuJoCo loop.
- [ ] Genesis feature analysis.

See **NEXT_STEPS.md** for the ordered task list.
