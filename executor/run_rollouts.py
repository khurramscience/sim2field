"""
Sim2Field — Multi-Sim Executor
===============================
--mock : synthesize plausible rollouts so the Phase A pipeline runs with no
         MuJoCo/GPU (pre-recorded-style episodes for the report + metrics).
real   : TODO wire MolmoAct2 / DINO-WM into the MuJoCo/MJX step loop (Phase B).

Each rollout conforms to rollout_record in schemas/scenario.schema.json.
Difficulty per axis/severity drives per-policy success probability so the
generated leaderboard and heatmap are realistic and discriminative.
"""
from __future__ import annotations
import argparse, json, os, sys, random

# relative policy competence (0..1) — MolmoAct2 best, scripted floor worst
POLICY_SKILL = {"molmoact2":0.92, "dino-wm":0.84, "pi05":0.78, "scripted":0.55}
# how hard each axis is, and how much each policy's skill is discounted on it
AXIS_HARDNESS = {"spatial":0.06,"physics":0.20,"dynamics":0.30,
                 "perception":0.16,"language":0.05,"distractor":0.24}
SEV_MULT = {"easy":0.4,"medium":1.0,"hard":1.6}

def _run(policy, grid, seed=0):
    rng = random.Random(hash((policy,seed)) & 0xffffffff)
    skill = POLICY_SKILL.get(policy,0.5)
    out = []
    for s in grid["scenarios"]:
        hard = AXIS_HARDNESS[s["axis"]] * SEV_MULT[s["severity"]]
        # batch-action policies (molmoact2) take an extra hit on dynamics events
        if policy=="molmoact2" and s["axis"]=="dynamics": hard += 0.16
        for k in range(s["success_criterion"].get("seeds",5)):
            r = random.Random(hash((policy,s["id"],k)) & 0xffffffff)
            p_succ = max(0.02, min(0.98, skill - hard + r.uniform(-0.08,0.08)))
            success = r.random() < p_succ
            # uncertainty trace: low when confident, spikes before failure
            T = 60
            base = (1-p_succ)*0.25 + r.uniform(-0.05,0.05)
            trace = [round(max(0,base) + 0.06*r.random(),3) for _ in range(T)]
            if success and r.random() < 0.20:            # nervous-but-fine successes
                sp=r.randint(10,T-8)
                for j in range(sp,sp+5): trace[j]=round(min(1.0,trace[j]+r.uniform(0.4,0.7)),3)
            ftime = None; fmode = None
            if not success:
                ft = r.randint(25, T-5)
                if r.random() > 0.30:                     # 70% of failures flagged, 30% silent
                    bump = r.uniform(0.45,0.9)
                    for j in range(max(0,ft-6), ft):
                        trace[j] = round(min(1.0, trace[j] + (j-(ft-6))*0.02*bump*10),3)
                ftime = round(ft/10.0,2)
                fmode = {"physics":"grasp_slip_drop","dynamics":"grasp_slip_drop",
                         "perception":"missed_affordance","distractor":"missed_affordance",
                         "spatial":"missed_affordance","language":"wrong_target"}[s["axis"]]
            out.append({"policy":policy,"scenario_id":s["id"],"axis":s["axis"],
                        "severity":s["severity"],"seed":k,"success":success,
                        "time_to_success_s": round(r.uniform(2,5),2) if success else None,
                        "failure_time_s":ftime,"failure_mode":fmode,
                        "uncertainty_trace":trace,
                        "cross_sim":{"mujoco":success,
                                     "genesis": success if r.random()<0.85 else (not success)}})
    return out

def _run_real(policy, grid, scenes_dir, seeds, scenario_id=None):
    """Phase B: step the policy in a real MuJoCo loop for one (or every) scenario
    that has a World-Builder MJCF + manifest. Falls back to nothing if a scene
    hasn't been built. Records carry source='mujoco_real'."""
    from executor.real_mujoco import run_real_rollout
    scens = grid["scenarios"]
    if scenario_id:
        scens = [s for s in scens if s["id"] == scenario_id]
        if not scens:
            sys.exit(f"[executor] scenario '{scenario_id}' not in grid")
    out = []
    for s in scens:
        sid = s["id"]
        mjcf = os.path.join(scenes_dir, f"{sid}.xml")
        man = os.path.join(scenes_dir, f"{sid}.manifest.json")
        if not (os.path.exists(mjcf) and os.path.exists(man)):
            print(f"[executor] skip {sid}: no MJCF/manifest in {scenes_dir} "
                  f"(run worldbuilder/build_scene.py first)", file=sys.stderr)
            continue
        manifest = json.load(open(man))
        for k in range(seeds):
            out.append(run_real_rollout(policy, manifest, mjcf, seed=k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--mock", action="store_true", default=True)
    ap.add_argument("--real", action="store_true",
                    help="Phase B: step the policy in a real MuJoCo loop (needs built scenes)")
    ap.add_argument("--scenes", default="data/scenes",
                    help="dir of World-Builder MJCF+manifest (for --real)")
    ap.add_argument("--scenario", help="single scenario id (for --real)")
    ap.add_argument("--out", default="data/rollouts")
    a = ap.parse_args()
    grid = json.load(open(a.grid))
    os.makedirs(a.out, exist_ok=True)

    if a.real:
        rollouts = _run_real(a.policy, grid, a.scenes, a.seeds, a.scenario)
        if not rollouts:
            sys.exit("[executor] --real produced no rollouts (build scenes first)")
        tag = "real"
    else:
        rollouts = _run(a.policy, grid)
        tag = "mock"

    path = os.path.join(a.out, f"{a.policy}.json")
    json.dump(rollouts, open(path, "w"))
    succ = sum(r["success"] for r in rollouts) / len(rollouts)
    print(f"[executor] {a.policy} ({tag}): {len(rollouts)} rollouts -> {path}  (success {succ:.0%})")

if __name__ == "__main__":
    main()
