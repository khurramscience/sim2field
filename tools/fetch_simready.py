#!/usr/bin/env python3
"""
Fetch a REAL imagine.io SimReady-Kitchens scene and convert it to a Sim2Field
scene.json (the input the Scenario Planner + World Builder consume).

The dataset is GATED (CC-BY-NC-4.0). You must:
  1) accept the terms at
     https://huggingface.co/datasets/imagineio/PhysicalAI-SimReady-Kitchens-v1
  2) provide a token — either `huggingface-cli login`, or HF_TOKEN env var,
     or --token on the command line.

Usage:
    # list available scene ids (uses the public file index; no token needed)
    python3 tools/fetch_simready.py --list | head

    # download ONE scene's lightweight files and convert -> data/galley_real.json
    python3 tools/fetch_simready.py --scene var_galley_0137d9bc --out data/galley_real.json

    # let it pick the Nth scene (matches the dataset's own folder ordering)
    python3 tools/fetch_simready.py --index 10 --out data/galley_real.json

Then run the pipeline on the real scene:
    ./demo.sh                      # (after: SCENE=data/galley_real.json)
    SCENE=data/galley_real.json ./demo.sh
"""
from __future__ import annotations
import argparse, json, os, re, sys

REPO = "imagineio/PhysicalAI-SimReady-Kitchens-v1"

# Map SimReady semantic labels -> our affordance types. Extend as needed; the
# converter falls back to "graspable" for anything movable it doesn't recognize.
LABEL_TO_TYPE = {
    "cabinet": "hinge", "door": "hinge", "fridge": "hinge", "refrigerator": "hinge",
    "oven": "hinge", "dishwasher": "hinge", "microwave": "hinge",
    "drawer": "slider", "faucet": "slider", "tap": "slider",
    "knob": "button", "switch": "button", "button": "button", "handle": "hinge",
    "sink": "cavity", "basin": "cavity", "pot": "cavity", "pan": "cavity",
    "bottle": "graspable", "mug": "graspable", "cup": "graspable", "kettle": "graspable",
    "plate": "graspable", "bowl": "graspable", "glass": "graspable", "utensil": "graspable",
    "counter": "surface", "countertop": "surface", "shelf": "surface", "table": "surface",
}


def _list_scene_ids():
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    ids = sorted({m.group(1) for f in files
                  for m in [re.match(r"scenes/([^/]+)/", f)] if m})
    return ids


def _classify(label: str) -> str:
    l = (label or "").lower()
    for key, typ in LABEL_TO_TYPE.items():
        if key in l:
            return typ
    return "graspable"


def _download_scene(scene_id: str, token: str | None):
    """Download just the lightweight metadata/annotation files for one scene."""
    from huggingface_hub import snapshot_download
    patterns = [
        f"scenes/{scene_id}/{scene_id}_metadata.json",
        f"scenes/{scene_id}/{scene_id}.png",
        f"scenes/{scene_id}/annotations/bounding_box_3d_0000.npy",
        f"scenes/{scene_id}/annotations/bounding_box_3d_labels_0000.json",
        f"scenes/{scene_id}/annotations/bounding_box_3d_prim_paths_0000.json",
        f"scenes/{scene_id}/annotations/metadata.txt",
        f"scenes/{scene_id}/annotations/manifest.json",
    ]
    path = snapshot_download(repo_id=REPO, repo_type="dataset",
                             allow_patterns=patterns, token=token)
    return os.path.join(path, "scenes", scene_id)


def _affordances_from_bbox3d(scene_dir: str):
    """Build affordance list from the 3D bounding boxes + labels (real poses)."""
    import numpy as np
    ann = os.path.join(scene_dir, "annotations")
    bbox = np.load(os.path.join(ann, "bounding_box_3d_0000.npy"), allow_pickle=True)
    labels = json.load(open(os.path.join(ann, "bounding_box_3d_labels_0000.json")))
    # labels is typically {idx: {"class": name}} or a list; normalize to list
    if isinstance(labels, dict):
        labels = [labels[k].get("class", k) if isinstance(labels[k], dict) else labels[k]
                  for k in sorted(labels, key=lambda x: int(x) if str(x).isdigit() else 0)]
    affs, seen = [], {}
    for i, row in enumerate(bbox):
        # USD bbox rows carry an axis-aligned box; center = mean of min/max corners.
        try:
            vals = [float(v) for v in np.asarray(row).ravel() if np.isscalar(v) or np.ndim(v) == 0]
            xs = vals[:6]
            cx, cy, cz = (xs[0] + xs[3]) / 2, (xs[1] + xs[4]) / 2, (xs[2] + xs[5]) / 2
        except Exception:
            continue
        label = labels[i] if i < len(labels) else f"object_{i}"
        base = re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_") or f"object_{i}"
        seen[base] = seen.get(base, 0) + 1
        name = base if seen[base] == 1 else f"{base}_{seen[base]}"
        affs.append({"name": name, "type": _classify(label),
                     "pose": [round(cx, 3), round(cy, 3), round(cz, 3), 0, 0, 0, 1]})
    return affs


def convert(scene_id: str, scene_dir: str, out_path: str):
    summary = {"domain": "kitchen", "layout": "galley"}
    try:
        meta = json.load(open(os.path.join(scene_dir, f"{scene_id}_metadata.json")))
        summary.update({k: meta[k] for k in ("layout", "category", "style") if k in meta})
    except Exception:
        meta = {}
    try:
        affs = _affordances_from_bbox3d(scene_dir)
    except Exception as e:
        print(f"[fetch] bbox parse failed ({e}); writing scene with empty affordances",
              file=sys.stderr)
        affs = []
    summary["surfaces"] = sum(1 for a in affs if a["type"] in ("surface", "cavity")) or 6
    summary["affordances"] = affs
    scene = {
        "scene_id": scene_id,
        "source": {
            "dataset": REPO,
            "url": f"https://huggingface.co/datasets/{REPO}",
            "library": "https://simready-kitchens.imagine.io/",
            "typology": summary.get("layout", "galley"),
            "license": "cc-by-nc-4.0",
            "note": "Affordances derived from the scene's real 3D bounding boxes + labels.",
        },
        "scene_summary": summary,
    }
    json.dump(scene, open(out_path, "w"), indent=2)
    print(f"[fetch] wrote {out_path}  ({len(affs)} affordances from real bbox3d)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list scene ids (public, no token)")
    ap.add_argument("--scene", help="scene id, e.g. var_galley_0137d9bc")
    ap.add_argument("--index", type=int, help="pick the Nth scene id instead of --scene")
    ap.add_argument("--out", default="data/galley_real.json")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF token (or set HF_TOKEN / huggingface-cli login)")
    a = ap.parse_args()

    if a.list:
        for s in _list_scene_ids():
            print(s)
        return

    scene_id = a.scene
    if a.index is not None:
        scene_id = _list_scene_ids()[a.index]
    if not scene_id:
        sys.exit("provide --scene <id> or --index <n> (or --list)")

    try:
        scene_dir = _download_scene(scene_id, a.token)
    except Exception as e:
        name = type(e).__name__
        if "Gated" in name or "401" in str(e) or "403" in str(e):
            sys.exit(f"[fetch] {scene_id}: dataset is gated. Accept terms at "
                     f"https://huggingface.co/datasets/{REPO} and provide a token "
                     f"(HF_TOKEN=... or huggingface-cli login).")
        raise
    convert(scene_id, scene_dir, a.out)


if __name__ == "__main__":
    main()
