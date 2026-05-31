# Sim2Field

**Scan the field. See it fail. Fix it before it ships.**

*Sim Eval Benchmark Creator — Google DeepMind Enterprise Build Day 2026*

---

Sim2Field turns a single scan of a real deployment site into a diverse, adversarial,
physics-grounded evaluation suite, runs candidate robot policies across every generated
variation, and returns a deployment report that predicts **where, how, and with what
confidence each policy will fail on your actual floor** — before the robot touches hardware.

The deliverable a robotics buyer wants isn't a benchmark score. It's: *"here are the 7
situations on your line where this policy drops the box, and here's the one policy that
doesn't."*

## Why this exists

Vision-Language-Action policies memorize spatial layouts and collapse when the world is
reconfigured (the affordance-generalization gap). Pure simulators can't be trusted because
sim-to-real friction is unsolved. Pure teleop data factories don't tell you what breaks.

Sim2Field doesn't claim a perfect simulator. It is the **failure-discovery and ranking
layer**: its value is measured by *discrimination* (does it separate policies?),
*transfer* (does its ranking match reality?), and *calibration* (does uncertainty predict
failure?) — not by visual realism.

## How it works

```
 Scan site ──▶ Scenario Planner ──▶ World Builder ──▶ Multi-Sim Executor ──▶ Scorer ──▶ Arena + Report
 (mobile)      (Gemini authors      (Genie 3 scenery   (MuJoCo physics-truth   (success,    (leaderboard,
               perturbation grid    + Veo renders +    + Genesis 2nd opinion)  uncertainty, failure heatmap,
               + success criteria)  scanned assets)                            failure mode) tap-to-inspect)
```

### The six perturbation axes
The Scenario Planner generates variations along six failure axes, each with a tunable knob:

1. **Spatial reconfiguration** — reposition affordances/objects (BusyBox-style).
2. **Physics** — sweep friction, mass, restitution (we *sweep* the unknown, not guess it).
3. **Dynamics events** — slip, drop, external push (catches low-frequency-policy failures).
4. **Perception** — lighting, texture, camera pose, sensor noise, partial observability.
5. **Language** — paraphrase the task instruction.
6. **Distractors** — clutter and novel objects.

## Google / DeepMind stack

| Component | Role |
|---|---|
| **Gemini API** | Scenario Planner — reads the scan, authors the perturbation grid + success criteria |
| **Genie 3 / Project Genie** | Scenery & layout variation (navigable scene variants from an image)¹ |
| **Veo** | Photorealistic renders for the perception axis + demo reel |
| **MuJoCo + MJX / MuJoCo Playground** | GPU-scaled physics-truth execution |
| **Open X-Embodiment** | Real robot trajectories for grounding and baselines |
| **Google Scanned Objects** | 3D assets for the distractor / novel-object axis |
| **Street View scenery** | Real-world site grounding for large/outdoor scenes |

¹ Genie 3 is navigation-focused and not a physics engine, so it drives visual/layout
diversity while MuJoCo owns contact dynamics. This split *is* the "combine generation with
good physics" design.

## Policies under test (all open, reproducible)

- **MolmoAct 2** (Ai2) — open Action Reasoning Model, evaluated on SimplerEnv / LIBERO.
- **DINO-WM** — world model on DINOv2 features, zero-shot MPC planning.
- **π0.5** (open checkpoint) — VLA baseline.
- **Scripted floor** — hand-tuned heuristic as the discrimination floor.

## What we measure

- **Discrimination** — success-rate spread across policies per axis.
- **Sim→real rank transfer** — Spearman ρ between Sim2Field ranking and held-out real
  ranking, using **Microsoft BusyBox** recorded episodes as ground truth. *(Headline number.)*
- **Calibration** — AUROC of per-step model uncertainty predicting episode failure.
- **Cross-sim agreement** — where MuJoCo and Genesis disagree, and which tracks reality.

## Build phases

- **Phase A (must-ship):** generation + arena + report on **pre-recorded rollouts** —
  policies run for real, once, offline; report reads from logged episodes. De-risks the demo.
- **Phase B (upside):** **live closed-loop** policy execution inside the MuJoCo step loop,
  Genesis cross-check, Veo photoreal cards.

## Repository layout

```
planner/        Gemini Scenario Planner — scan -> perturbation grid (JSON)
worldbuilder/   Genie 3 + Veo + Scanned Objects -> sim-ready scene variants
executor/       MuJoCo / MJX rollout runner; logs HDF5 episodes (BusyBox-compatible)
scorer/         success, uncertainty, failure-mode, sim->real correlation, calibration
app/            mobile web app (scan -> envs -> arena -> report -> inspect)
schemas/        scenario + rollout record JSON schemas
data/           cached scans, generated scenarios, rollouts
```

## Quickstart (Phase A)

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...                       # Scenario Planner
python planner/generate_scenarios.py --scene data/kitchen.json --out data/grid.json
python executor/run_rollouts.py  --grid data/grid.json --policy molmoact2 --seeds 5
python scorer/build_report.py    --rollouts data/rollouts/ --truth data/busybox/
# open app/index.html on your phone
```

## Demo

Mobile prototype (clickable): `app/index.html`

## License

MIT — see `LICENSE`.
