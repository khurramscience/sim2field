# Sim2Field — 60-second demo

**Scan the field. See it fail. Fix it before it ships.**

One command rebuilds the whole pipeline from a real **imagine.io SimReady-Kitchens**
galley scene and serves the mobile app:

```bash
./demo.sh
# open http://localhost:8000/app/index.html  (phone: same URL, this machine's IP)
```

To author the perturbation grid **live with Gemini 3.5** instead of the offline fallback:

```bash
GEMINI_API_KEY=...  ./demo.sh
# pin a specific model:  GEMINI_MODEL=gemini-3.5-pro GEMINI_API_KEY=... ./demo.sh
```

To pull a **real (gated) SimReady scene** and run on it (needs an HF token with the
dataset terms accepted):

```bash
# one-off: download + convert a real scene, then run the demo on it
HF_TOKEN=hf_xxx  FETCH_SCENE=var_galley_0137d9bc  ./demo.sh

# or just fetch + convert (browse ids first):
python3 tools/fetch_simready.py --list | head
HF_TOKEN=hf_xxx python3 tools/fetch_simready.py --index 10 --out data/galley_real.json
SCENE=data/galley_real.json ./demo.sh
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
| **imagine.io SimReady-Kitchens** | Real dataset (800 galley scenes). `data/simready/catalog.json` is the real scene list (public HF API). The assets are **gated** (CC-BY-NC) — `tools/fetch_simready.py` downloads a real scene and converts its 3D bounding boxes → affordances **when you supply an HF token** with terms accepted; `data/galley.json` is the committed offline scene used when no token is present. |
| **MuJoCo** | Real. All 24 generated scenes load and step in MuJoCo 3.x; `--real` decides success from physics. |
| **Gemini planner** | Real API call (`google-genai`); runs live with `GEMINI_API_KEY` (tries `gemini-3.5-pro` first, override via `GEMINI_MODEL`), otherwise a deterministic, scene-aware fallback. Provenance is in `grid._provenance.planner`. |
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
