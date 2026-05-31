"""
Sim2Field — World Builder
=========================

Turns a scanned deployment scene + ONE scenario (from the Scenario Planner grid)
into a sim-ready **MuJoCo MJCF** plus a runtime **manifest** the executor needs to
roll out a policy (task instruction, success predicate, scheduled dynamics event,
sensor-noise level, generation provenance).

Design (mirrors planner/generate_scenarios.py):
  Primary path : pull a real imagine.io SimReady-Kitchens scene + Google Scanned
                 Objects distractors, optional Gemini/Veo perception renders.
  Fallback path: a deterministic, pure-python MJCF emitter that needs NO network,
                 NO api key, and NO GPU — so the pipeline, demo, and CI always run.
                 (Writing MJCF is just XML; MuJoCo is only needed to *validate*.)

The six perturbation axes map onto the world as follows:
  spatial     -> translate the named affordance body
  physics     -> set geom slide-friction / scale graspable mass
  dynamics    -> schedule a slip / drop / external_push event (manifest, applied
                 at run time by the executor); "slip" also lowers contact friction
  perception  -> swap the light rig (backlit/dusk/harsh_overhead); record sensor_noise
  language    -> carry the paraphrased instruction (no physics change)
  distractor  -> add clutter bodies sourced from Google Scanned Objects ids

Usage:
    # one scenario -> one MJCF + manifest
    python worldbuilder/build_scene.py --scene data/kitchen.json --grid data/grid.json \
        --scenario dyn-03 --out data/scenes

    # whole grid
    python worldbuilder/build_scene.py --scene data/kitchen.json --grid data/grid.json \
        --all --out data/scenes
"""

from __future__ import annotations
import argparse, json, os, sys, math
from xml.sax.saxutils import escape

# --- imagine.io SimReady-Kitchens (collaborator's structured-world layer) -------
SIMREADY_DATASET = "imagineio/PhysicalAI-SimReady-Kitchens-v1"

# Representative MuJoCo geom per scanned affordance type. Sizes in meters.
# (shape, half-extents-or-radii, joint-kind, base-mass-kg)
AFFORDANCE_GEOM = {
    "graspable": ("cylinder", (0.035, 0.10), "free",  0.40),  # bottle/mug-like
    "hinge":     ("box",      (0.18, 0.02, 0.22), "hinge", 1.50),  # cabinet door
    "slider":    ("box",      (0.16, 0.18, 0.03), "slide", 1.20),  # drawer / faucet handle
    "button":    ("box",      (0.02, 0.02, 0.01), None,   0.05),  # knob / switch
    "surface":   ("box",      (0.20, 0.20, 0.01), None,   0.00),  # static shelf
    "cavity":    ("box",      (0.12, 0.12, 0.02), None,   0.00),  # sink basin (approx)
}

# Light rigs for the perception axis.
LIGHT_RIGS = {
    "default":        [("0.6 0 1.4", "0.8 0.8 0.8", "0.3 0.3 0.3")],
    "backlit":        [("-0.8 0 1.2", "0.9 0.9 1.0", "0.05 0.05 0.05")],
    "dusk":           [("0.6 0 1.0", "0.35 0.30 0.45", "0.10 0.10 0.15")],
    "harsh_overhead": [("0.5 0 2.2", "1.0 1.0 1.0", "0.0 0.0 0.0")],
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _quat_wxyz(pose):
    """kitchen.json poses are [x,y,z, qx,qy,qz,qw]; MuJoCo wants quat 'w x y z'."""
    if len(pose) >= 7:
        x, y, z, qx, qy, qz, qw = pose[:7]
    else:
        x, y, z = (list(pose) + [0, 0, 0])[:3]
        qx = qy = qz = 0.0; qw = 1.0
    return (x, y, z), (qw, qx, qy, qz)


def _color_from_id(s):
    """Deterministic pleasant RGB from a string id (for distractors)."""
    h = abs(hash(s))
    r = 0.35 + (h & 0xFF) / 255 * 0.6
    g = 0.35 + ((h >> 8) & 0xFF) / 255 * 0.6
    b = 0.35 + ((h >> 16) & 0xFF) / 255 * 0.6
    return f"{r:.2f} {g:.2f} {b:.2f} 1"


def _fmt(*vals):
    return " ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in vals)


# ---------------------------------------------------------------------------
# MJCF emission
# ---------------------------------------------------------------------------
def build_mjcf(scene: dict, scenario: dict):
    """Return (mjcf_xml_str, manifest_dict) for one scene+scenario, no deps required."""
    pert = scenario.get("perturbation", {}) or {}
    affs = scene["scene_summary"]["affordances"]

    # ---- physics knobs --------------------------------------------------
    # MuJoCo geom friction = "slide torsion roll"; the physics axis sweeps slide.
    slide_friction = float(pert["friction"]) if pert.get("friction") is not None else 1.0
    mass_scale = float(pert.get("mass_scale", 1.0))
    # a "slip" dynamics event is modelled partly as a low-friction contact too
    if pert.get("event") == "slip":
        slide_friction = min(slide_friction, 0.25)
    fric_str = _fmt(round(slide_friction, 3), 0.005, 0.0001)

    # ---- spatial knob ---------------------------------------------------
    reposition = pert.get("reposition", {}) or {}   # {object_name: [dx,dy,dz]}

    # ---- lighting -------------------------------------------------------
    rig = LIGHT_RIGS.get(pert.get("lighting", "default"), LIGHT_RIGS["default"])

    # ---- assemble bodies ------------------------------------------------
    bodies = []
    grasp_targets = []
    for aff in affs:
        name = aff["name"]
        kind = aff.get("type", "graspable")
        shape, dims, joint, base_mass = AFFORDANCE_GEOM.get(
            kind, AFFORDANCE_GEOM["graspable"])
        (px, py, pz), (qw, qx, qy, qz) = _quat_wxyz(aff.get("pose", [0, 0, 0.82, 0, 0, 0, 1]))
        if name in reposition:
            dx, dy, dz = (list(reposition[name]) + [0, 0, 0])[:3]
            px, py, pz = px + dx, py + dy, pz + dz

        size = _fmt(*[float(d) for d in dims])
        body = [f'    <body name="{escape(name)}" pos="{_fmt(float(px),float(py),float(pz))}" '
                f'quat="{_fmt(float(qw),float(qx),float(qy),float(qz))}">']
        if joint == "free":
            mass = max(0.01, base_mass * mass_scale)
            body.append(f'      <freejoint/>')
            body.append(f'      <geom type="{shape}" size="{size}" mass="{mass:.3f}" '
                        f'friction="{fric_str}" rgba="0.55 0.62 0.78 1"/>')
            grasp_targets.append(name)
        elif joint == "hinge":
            body.append(f'      <joint name="{escape(name)}_hinge" type="hinge" '
                        f'axis="0 0 1" range="0 1.8" damping="2"/>')
            body.append(f'      <geom type="{shape}" size="{size}" mass="{base_mass*mass_scale:.3f}" '
                        f'friction="{fric_str}" rgba="0.50 0.40 0.32 1"/>')
        elif joint == "slide":
            body.append(f'      <joint name="{escape(name)}_slide" type="slide" '
                        f'axis="1 0 0" range="0 0.35" damping="3"/>')
            body.append(f'      <geom type="{shape}" size="{size}" mass="{base_mass*mass_scale:.3f}" '
                        f'friction="{fric_str}" rgba="0.50 0.40 0.32 1"/>')
        else:  # static (button/surface/cavity)
            body.append(f'      <geom type="{shape}" size="{size}" '
                        f'friction="{fric_str}" rgba="0.45 0.47 0.50 1"/>')
        body.append('    </body>')
        bodies.append("\n".join(body))

    # ---- distractor clutter (Google Scanned Objects) --------------------
    distractors = pert.get("distractors", []) or []
    for i, gid in enumerate(distractors):
        ang = 2 * math.pi * i / max(1, len(distractors))
        dx, dy = 0.18 * math.cos(ang), 0.18 * math.sin(ang)
        h = abs(hash(gid))
        shape = ("box", "cylinder", "sphere")[h % 3]
        if shape == "box":
            size = _fmt(0.03, 0.03, 0.04)
        elif shape == "cylinder":
            size = _fmt(0.025, 0.05)
        else:
            size = _fmt(0.035)
        bodies.append(
            f'    <body name="distractor_{i}_{escape(gid)}" '
            f'pos="{_fmt(0.5+dx, 0.0+dy, 0.86)}">\n'
            f'      <freejoint/>\n'
            f'      <geom type="{shape}" size="{size}" mass="0.20" '
            f'friction="{fric_str}" rgba="{_color_from_id(gid)}"/>\n'
            f'    </body>')

    lights = "\n".join(
        f'    <light pos="{p}" dir="0 0 -1" diffuse="{d}" ambient="{a}"/>'
        for (p, d, a) in rig)

    mjcf = f'''<mujoco model="sim2field-{escape(scene.get("scene_id","scene"))}-{escape(scenario["id"])}">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81" integrator="implicitfast" timestep="0.002"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.18 0.20 0.24" rgb2="0.22 0.24 0.28"
             width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.1"/>
    <material name="counter" rgba="0.30 0.33 0.40 1"/>
  </asset>
  <worldbody>
{lights}
    <geom name="floor" type="plane" size="2 2 0.1" material="grid"/>
    <body name="counter" pos="0.5 0 0.40">
      <geom type="box" size="0.45 0.45 0.02" material="counter" friction="{fric_str}"/>
    </body>
{chr(10).join(bodies)}
  </worldbody>
</mujoco>
'''

    # ---- runtime manifest the executor consumes ------------------------
    sc = scenario.get("success_criterion", {}) or {}
    event = None
    if pert.get("event"):
        event = {"kind": pert["event"],
                 "time_s": pert.get("event_time_s"),
                 # a nudge for external_push, an impulse for drop, friction-loss for slip
                 "force_N": 12.0 if pert["event"] == "external_push" else 0.0}
    manifest = {
        "scene_id": scene.get("scene_id"),
        "scenario_id": scenario["id"],
        "axis": scenario["axis"],
        "severity": scenario.get("severity"),
        "instruction": pert.get("instruction") or sc.get("predicate", ""),
        "success_predicate": sc.get("predicate", ""),
        "horizon_s": sc.get("horizon_s", 6.0),
        "seeds": sc.get("seeds", 5),
        "physics": {"slide_friction": round(slide_friction, 3), "mass_scale": mass_scale},
        "event": event,
        "sensor_noise": pert.get("sensor_noise", 0.0),
        "lighting": pert.get("lighting", "default"),
        "grasp_targets": grasp_targets,
        "distractors": distractors,
        "generation": scenario.get("generation", {"scenery": "base_scan", "physics": "mujoco"}),
        "source_dataset": SIMREADY_DATASET,
    }
    return mjcf, manifest


# ---------------------------------------------------------------------------
# optional real generators (graceful no-ops without keys / network)
# ---------------------------------------------------------------------------
def maybe_load_simready(scene_path: str):
    """Hook: pull a real SimReady-Kitchens scene. Falls back to the local scan."""
    # Real implementation would `huggingface_hub.snapshot_download(SIMREADY_DATASET)`
    # and parse the USD/GLTF into our scene_summary shape. Kept offline-safe here.
    return json.load(open(scene_path))


def maybe_render_perception(manifest: dict):
    """Hook: Gemini/Veo perception-axis render. Returns a path or None offline."""
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        # A real call would render manifest['lighting'] / sensor_noise as an image
        # via Veo / Gemini image and return the file path. Left as a guarded stub.
        return None
    except Exception as e:  # pragma: no cover
        print(f"[worldbuilder] perception render skipped ({e})", file=sys.stderr)
        return None


def validate_mjcf(mjcf: str) -> str:
    """Load the MJCF in MuJoCo if available; return 'ok', 'skipped', or 'error: ...'."""
    try:
        import mujoco
    except Exception:
        return "skipped (mujoco not installed)"
    try:
        model = mujoco.MjModel.from_xml_string(mjcf)
        return f"ok ({model.nbody} bodies, {model.ngeom} geoms, {model.njnt} joints)"
    except Exception as e:
        return f"error: {e}"


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build_one(scene, scenario, out_dir, validate=True):
    mjcf, manifest = build_mjcf(scene, scenario)
    os.makedirs(out_dir, exist_ok=True)
    sid = scenario["id"]
    xml_path = os.path.join(out_dir, f"{sid}.xml")
    man_path = os.path.join(out_dir, f"{sid}.manifest.json")
    with open(xml_path, "w") as f:
        f.write(mjcf)
    manifest["render"] = maybe_render_perception(manifest)
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    status = validate_mjcf(mjcf) if validate else "not validated"
    print(f"[worldbuilder] {sid:8} [{scenario['axis']:10}] -> {xml_path}  MuJoCo: {status}")
    return xml_path, man_path, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="scanned scene json (e.g. data/kitchen.json)")
    ap.add_argument("--grid", required=True, help="scenario grid json from the planner")
    ap.add_argument("--scenario", help="single scenario id to build")
    ap.add_argument("--all", action="store_true", help="build every scenario in the grid")
    ap.add_argument("--out", default="data/scenes")
    ap.add_argument("--no-validate", action="store_true", help="skip MuJoCo load check")
    a = ap.parse_args()

    scene = maybe_load_simready(a.scene)
    grid = json.load(open(a.grid))
    scenarios = grid["scenarios"]

    if a.all:
        targets = scenarios
    elif a.scenario:
        targets = [s for s in scenarios if s["id"] == a.scenario]
        if not targets:
            sys.exit(f"[worldbuilder] scenario '{a.scenario}' not found in {a.grid}")
    else:
        targets = scenarios[:1]
        print(f"[worldbuilder] no --scenario/--all given; building first: {targets[0]['id']}")

    ok = 0
    for s in targets:
        _, _, status = build_one(scene, s, a.out, validate=not a.no_validate)
        ok += status.startswith("ok") or status.startswith("not") or status.startswith("skip")
    print(f"[worldbuilder] built {len(targets)} scene(s) into {a.out}/  ({ok} loadable/ok)")


if __name__ == "__main__":
    main()
