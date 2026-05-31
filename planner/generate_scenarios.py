"""
Sim2Field — Scenario Planner
=============================

Turns a scanned deployment scene into a grid of perturbed RL environments
(JSON conforming to schemas/scenario.schema.json).

Primary path : Gemini authors the grid (creative, scene-aware hard cases).
Fallback path: a deterministic generator runs with NO api key / NO network,
               so the pipeline, demo, and CI never depend on a live model.

Usage:
    python planner/generate_scenarios.py --scene data/kitchen.json --out data/grid.json --n 24
"""

from __future__ import annotations
import argparse, json, os, sys, random

AXES = ["spatial", "physics", "dynamics", "perception", "language", "distractor"]

SYSTEM_PROMPT = """\
You are Sim2Field's Scenario Planner, an expert robotics test engineer.

You are given a SCENE captured at a real robot deployment site: its domain, its
surfaces, and its affordances (graspable objects, hinges, sliders, buttons,
cavities) with 3D poses. Author a GRID of evaluation scenarios that EXPOSE how a
manipulation policy fails on THIS specific floor, before deployment.

Design principles:
- Adversarial but physically plausible. Target known VLA failure modes: spatial
  memorization, low grip force under low friction, slow re-planning during fast
  events, sensitivity to lighting/viewpoint, brittle language grounding, and
  distraction by novel objects.
- Ground every scenario in the scene's real affordances and poses.
- Cover all six axes: spatial, physics, dynamics, perception, language,
  distractor. Bias toward HARD cases; easy cases don't discriminate.
- Each scenario needs a crisp, machine-checkable success_criterion.predicate.
- Choose a generator per scenario: scenery in {genie3, veo, scanned_objects,
  base_scan}, physics in {mujoco, mjx, genesis}.

Return ONLY valid JSON matching the schema. No prose, no markdown.
"""

def _spatial(aff, rng):
    o = rng.choice(aff); dx = rng.choice([0.12, 0.24, -0.18, 0.30])
    return (f"{o['name']} moved {int(dx*100):+d}cm", rng.choice(["easy","medium"]),
            {"reposition": {o["name"]: [dx,0.0,0.0]}},
            {"scenery":"genie3","physics":"mujoco"},
            f"task succeeds with {o['name']} at new pose")

def _physics(aff, rng):
    if rng.random() < 0.5:
        mu = round(rng.uniform(0.22,0.38),2)
        return (f"Low-friction surface mu={mu}","hard",{"friction":mu},
                {"scenery":"base_scan","physics":"mujoco"},
                "object grasped and held without slip")
    m = round(rng.uniform(1.3,2.1),1); o = rng.choice(aff)
    return (f"Heavy {o['name']} x{m}","medium",{"mass_scale":m},
            {"scenery":"base_scan","physics":"mjx"}, f"{o['name']} lifted and placed")

def _dynamics(aff, rng):
    o = rng.choice(aff); ev = rng.choice(["slip","drop","external_push"]); t = round(rng.uniform(1.5,4.5),1)
    return (f"{o['name']} {ev} event @ {t}s","hard",{"event":ev,"event_time_s":t},
            {"scenery":"base_scan","physics":"mujoco"},
            f"policy recovers from {ev} and completes task")

def _perception(aff, rng):
    light = rng.choice(["backlit","dusk","harsh_overhead"]); noise = round(rng.uniform(0.05,0.2),2)
    return (f"{light} + cam noise {noise}", rng.choice(["medium","hard"]),
            {"lighting":light,"sensor_noise":noise},
            {"scenery":"veo","physics":"mujoco"}, "task succeeds under degraded perception")

def _language(aff, rng):
    o = rng.choice(aff); base = o['name'].split('_')[0]
    p = [f"grab the {base}", f"could you pick up that {base}", f"the {base} - move it"]
    ph = rng.choice(p)
    return (f'"{ph}"',"easy",{"instruction":ph},
            {"scenery":"base_scan","physics":"mujoco"}, f"policy grounds instruction to {o['name']}")

def _distractor(aff, rng):
    ids = rng.sample(["gso_mug_03","gso_can_11","gso_box_07","gso_bottle_22"], k=2)
    return ("Cluttered counter (novel objects)","hard",{"distractors":ids},
            {"scenery":"scanned_objects","physics":"mujoco"}, "target manipulated despite distractors")

_GEN = {"spatial":_spatial,"physics":_physics,"dynamics":_dynamics,
        "perception":_perception,"language":_language,"distractor":_distractor}

def _fallback_grid(scene, n, seed=7):
    rng = random.Random(seed)
    aff = scene["scene_summary"]["affordances"] or [{"name":"object","type":"graspable"}]
    weighted = ["spatial","physics","physics","dynamics","dynamics",
                "perception","perception","language","distractor","distractor"]
    scenarios = []
    for i in range(n):
        axis = weighted[i % len(weighted)]
        name, sev, pert, gen, pred = _GEN[axis](aff, rng)
        scenarios.append({"id":f"{axis[:3]}-{i:02d}","axis":axis,"name":name,"severity":sev,
                          "perturbation":pert,"generation":gen,
                          "success_criterion":{"predicate":pred,"horizon_s":6.0,"seeds":5}})
    return {"scene_id":scene["scene_id"],"scene_summary":scene["scene_summary"],"scenarios":scenarios}

def _gemini_grid(scene, n):
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key: return None
    # Use the newest available Gemini; fall back across model names so the call
    # works whichever generation the account/SDK exposes.
    model = os.environ.get("GEMINI_MODEL")
    candidates = [model] if model else ["gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-pro"]
    try:
        from google import genai
        client = genai.Client(api_key=key)
        user = f"SCHEMA: scenario.schema.json. Generate exactly {n} scenarios.\nSCENE:\n{json.dumps(scene,indent=2)}"
        last = None
        for m in [c for c in candidates if c]:
            try:
                resp = client.models.generate_content(
                    model=m,
                    config={"system_instruction": SYSTEM_PROMPT,
                            "response_mime_type": "application/json"},
                    contents=user)
                grid = json.loads(resp.text)
                grid.setdefault("_provenance", {})["planner"] = f"gemini:{m}"
                print(f"[planner] authored grid with {m}", file=sys.stderr)
                return grid
            except Exception as e:
                last = e
        raise last or RuntimeError("no gemini model available")
    except Exception as e:
        print(f"[planner] Gemini unavailable ({e}); using deterministic fallback.", file=sys.stderr)
        return None

def generate(scene, n=24):
    grid = _gemini_grid(scene, n)
    if grid is None:
        grid = _fallback_grid(scene, n)
        grid.setdefault("_provenance", {})["planner"] = "deterministic-fallback"
    # carry the source-dataset provenance from the scan onto the grid
    if isinstance(scene, dict) and scene.get("source"):
        grid.setdefault("_provenance", {})["scene_source"] = scene["source"]
    return grid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    scene = json.load(open(a.scene))
    grid = generate(scene, a.n)
    json.dump(grid, open(a.out,"w"), indent=2)
    axes = {}
    for s in grid["scenarios"]: axes[s["axis"]] = axes.get(s["axis"],0)+1
    print(f"[planner] wrote {len(grid['scenarios'])} scenarios to {a.out}")
    print("[planner] axis mix:", dict(sorted(axes.items())))

if __name__ == "__main__":
    main()
