# NEXT_STEPS — Sim2Field

Ordered task list for Claude Code. Each task is self-contained; do them top to bottom.

## 0. Push to GitHub (do this first)
The repo is ready. From this folder, logged in to GitHub:
```bash
./push.sh        # git init, commit, push to khurramscience/sim2field, then enable Pages
```
Then: GitHub → Settings → Pages → deploy from `main` / root.
Demo URL becomes: https://khurramscience.github.io/sim2field/app/index.html

## 1. Wire the mobile app to real pipeline output
Right now `app/index.html` shows hardcoded numbers. Make it load `data/report.json` so the
leaderboard, the policy×axis heatmap, and the inspect view reflect the actual pipeline run.
- Read `report.json` (policies, heatmap, calibration.auroc, sim2real.spearman_rho, cross_sim).
- Map heatmap success rates → the cell colors (pass/warn/fail thresholds, e.g. >0.6/0.3–0.6/<0.3).
- For inspect, read a chosen failing rollout's `uncertainty_trace` and draw the real sparkline.
- Keep it a single self-contained file (no localStorage; in-memory only).

## 2. Phase B — one real policy in MuJoCo (the credibility upgrade)
Replace the mock for a single scenario to prove the loop is real.
- Install: `pip install mujoco mujoco-mjx` (and the policy: MolmoAct 2 from Ai2 release, or DINO-WM).
- In `executor/run_rollouts.py`, add a `--real` path: load a SimReady kitchen, apply ONE
  perturbation (e.g. friction 0.28), step MuJoCo with the policy in the loop, log a real
  rollout_record (success, failure_time_s, failure_mode, uncertainty_trace, cross_sim).
- Start with one (policy × scenario × 3 seeds); keep the mock for the rest.

## 3. World Builder — implement `worldbuilder/build_scene.py`  ✅ DONE
`build_scene.py` turns the scene + one scenario into a **MuJoCo MJCF** plus a
runtime **manifest** the executor consumes. Pure-python emitter (no deps/network
required); validates with `mujoco` when installed.
- [x] Loads the scene (SimReady-Kitchens hook `maybe_load_simready`, offline-safe
      fallback to the local scan).
- [x] Applies each axis: spatial→translate body, physics→slide friction / mass
      scale, dynamics→scheduled slip/drop/push event (+ low friction for slip),
      perception→light rig swap + sensor_noise, language→carry instruction,
      distractor→clutter bodies from Google Scanned Objects ids.
- [x] Emits `<id>.xml` (MJCF) + `<id>.manifest.json` per scenario; `--scenario`,
      `--all`, `--no-validate` flags.
- [x] Gemini/Veo perception render + Genie 3 layout hooks present, guarded (no-op
      without `GEMINI_API_KEY`).
- Verified: all 24 scenes built and load in MuJoCo 3.9 (e.g. dyn-03 = 9 bodies /
  9 geoms / 4 joints) and step 250 physics steps with finite state.
- Output: `data/scenes/` (gitignored); two examples committed in
  `worldbuilder/examples/`.
- Run: `python worldbuilder/build_scene.py --scene data/kitchen.json --grid data/grid.json --all --out data/scenes`

## 4. Analyze Genesis for useful features (owner asked)
Repo: https://github.com/Genesis-Embodied-AI/genesis-world
- Assess: physics fidelity vs MuJoCo, GPU throughput, generative/scene features, asset pipeline.
- Note honestly where it's strong vs overstated (owner's prior: "possibly overhyped").
- Decide if it earns the "cross-check" slot or if Isaac/Newton fits better. Write findings to docs/.

## 5. Validation slide (the science punchline)
- Expand `data/busybox_truth.json` toward real BusyBox episode outcomes.
- Report Spearman ρ over MORE than 4 policies (4 makes ρ coarse). Add baselines if needed.
- Produce a one-figure result: sim ranking vs real ranking + calibration curve.

## Open thread
- An arxiv id the owner referenced (`2605.29710`) did not resolve — get the correct title/link
  before citing it as benchmark inspiration.
