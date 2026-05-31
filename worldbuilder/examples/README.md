# World Builder — example output

Two representative scenes produced by `worldbuilder/build_scene.py` from
`data/kitchen.json` + `data/grid.json`. The full grid (24 scenes) is generated
into `data/scenes/` (gitignored); these two are committed so the output is
visible without running anything.

Each scene is a pair:
- `*.xml` — a **MuJoCo MJCF** that loads and steps (validated with `mujoco`).
- `*.manifest.json` — the runtime contract the executor consumes (task
  instruction, success predicate, physics knobs, scheduled dynamics event,
  sensor noise, lighting, grasp targets, distractors, generation provenance).

| Scene | Axis | What the perturbation does |
|---|---|---|
| `dyn-03` | dynamics | `slip` event scheduled at t=1.8s; contact friction lowered to 0.25 |
| `dis-08` | distractor | adds clutter bodies sourced from Google Scanned Objects ids |

Regenerate everything:

```bash
python worldbuilder/build_scene.py --scene data/kitchen.json --grid data/grid.json --all --out data/scenes
# or a single scenario:
python worldbuilder/build_scene.py --scene data/kitchen.json --grid data/grid.json --scenario dyn-03 --out data/scenes
```

All 24 scenes load in MuJoCo (`mujoco 3.x`); without `mujoco` installed the
builder still emits valid MJCF and reports validation as skipped.
