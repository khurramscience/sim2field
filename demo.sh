#!/usr/bin/env bash
# Sim2Field — one-command demo.
# Regenerates the whole pipeline from a real imagine.io SimReady galley scene,
# then serves the mobile app so you can open it on your phone or browser.
#
#   ./demo.sh            # rebuild data + serve on :8000
#   ./demo.sh --serve    # just serve (use the committed data)
#   GEMINI_API_KEY=...    ./demo.sh   # author the grid live with Gemini
set -e
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
SCENE="${SCENE:-data/galley.json}"     # real SimReady galley scene (var_galley_0137d9bc)

if [ "$1" != "--serve" ]; then
  echo "▶ 1/4  Scenario Planner  — perturbations from $SCENE"
  python3 planner/generate_scenarios.py --scene "$SCENE" --out data/grid.json --n 24
  echo "▶ 2/4  Executor          — rollouts (mock, 3 seeds × 4 policies)"
  rm -f data/rollouts/*.json
  for p in molmoact2 dino-wm pi05 scripted; do
    python3 executor/run_rollouts.py --grid data/grid.json --policy "$p" --out data/rollouts --seeds 3
  done
  echo "▶ 3/4  Scorer            — report.json (success, heatmap, ρ, AUROC)"
  python3 scorer/build_report.py --rollouts data/rollouts --truth data/busybox_truth.json --out data/report.json
  echo "▶ 4/4  World Builder      — MuJoCo MJCF + manifest per scenario"
  python3 worldbuilder/build_scene.py --scene "$SCENE" --grid data/grid.json --all --out data/scenes | tail -1
  # refresh the app's embedded snapshot so file:// also shows the latest run
  python3 tools/embed_app_data.py
fi

echo
echo "✅  Open the demo:"
echo "     http://localhost:$PORT/app/index.html"
echo "     (phone: same URL with this machine's IP)"
echo
python3 -m http.server "$PORT"
