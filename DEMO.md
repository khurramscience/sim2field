# Sim2Field — 60-second demo

**Scan the field. See it fail. Fix it before it ships.**

One command rebuilds the whole pipeline from a real **imagine.io SimReady-Kitchens**
galley scene and serves the mobile app:

```bash
./demo.sh
# open http://localhost:8000/app/index.html  (phone: same URL, this machine's IP)
```

To author the perturbation grid **live with Gemini** instead of the offline fallback:

```bash
GEMINI_API_KEY=...  ./demo.sh
```

Already have data and just want the UI:

```bash
./demo.sh --serve
```

## What to show (the 6 screens)

1. **Scan** — the deployment site. The pills are real: scene
   `var_galley_0137d9bc`, *galley* layout, 8 surfaces, 10 affordances, **grounded
   in imagine.io SimReady-Kitchens**.
2. **Generate** — Gemini authors a 24-cell perturbation grid across the six
   failure axes (spatial, physics, dynamics, perception, language, distractor).
3. **Arena** — every policy run across all 24 environments; ranked leaderboard.
4. **Report** — the policy × axis **failure heatmap** (green pass / amber warn /
   red fail), plus sim→real Spearman ρ and calibration AUROC.
5. **Inspect** — tap any cell to see *why* a policy fails: the real
   `uncertainty_trace` as a sparkline, failure mode, timing, cross-sim agreement.
6. **Decision** — ship / don't-ship call for the best policy on this floor.

Every number in the app comes from the pipeline run (`data/report.json`,
`grid.json`, `rollouts/*.json`) — nothing is hardcoded.

## What's real vs. placeholder (we keep this honest)

| Piece | State |
|---|---|
| **imagine.io SimReady-Kitchens** | Real dataset (800 galley scenes); we ground in a real scene id. Asset meshes are gated (CC-BY-NC), so we use the scene's affordance layout, not the mesh files. `data/simready/catalog.json` is the real scene list pulled from the public HF API. |
| **MuJoCo** | Real. All 24 generated scenes load and step in MuJoCo 3.x. |
| **Gemini planner** | Real API call (`google-genai`); runs live with `GEMINI_API_KEY`, otherwise a deterministic, scene-aware fallback. Provenance is recorded in `grid._provenance.planner`. |
| **Rollouts / leaderboard** | Mock executor — realistic, discriminative **placeholders**, not real policy runs yet. |
| **Genie 3 / Veo / Scanned Objects** | Labeled provenance + guarded hooks; no live calls without keys. |

## Bonus: one REAL physics rollout (Phase B, no GPU)

Prove the loop isn't a mock — step a policy in **real MuJoCo physics** for one
scenario. The grip can only hold what Coulomb friction allows
(`max_hold = μ · squeeze`), so when friction or mass cross the limit the object
physically slips and drops:

```bash
python3 worldbuilder/build_scene.py --scene data/galley.json --grid data/grid.json --all --out data/scenes
# low-friction slip scenario -> the object really drops:
python3 -m executor.run_rollouts --policy scripted --real --grid data/grid.json --scenario dyn-03 --seeds 5
# adequate friction -> it holds:
python3 -m executor.run_rollouts --policy scripted --real --grid data/grid.json --scenario phy-01 --seeds 5
```

Observed: μ≈0.25–0.31 → 0% success (real drop), μ=1.0 → 100% — even at 1.4× mass.
Records are tagged `"source": "mujoco_real"`. `reaction_steps` in
`executor/real_mujoco.py` models batch-action / slow-replanning policies (the
seam where a real VLA like MolmoAct 2 plugs into the same loop). The leaderboard
in the app still uses the mock executor across all 24 scenarios; this is the
single-scenario credibility upgrade.

## Pipeline by hand

```bash
python3 planner/generate_scenarios.py --scene data/galley.json --out data/grid.json --n 24
for p in molmoact2 dino-wm pi05 scripted; do
  python3 executor/run_rollouts.py --grid data/grid.json --policy $p --out data/rollouts --seeds 3
done
python3 scorer/build_report.py --rollouts data/rollouts --truth data/busybox_truth.json --out data/report.json
python3 worldbuilder/build_scene.py --scene data/galley.json --grid data/grid.json --all --out data/scenes
python3 tools/embed_app_data.py        # refresh the app's offline snapshot
```
