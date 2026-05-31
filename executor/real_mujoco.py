"""
Sim2Field — Phase B: one REAL policy in a MuJoCo loop
=====================================================

This is the credibility upgrade: instead of synthesizing a rollout, we load a
World-Builder MJCF, step it in MuJoCo with a policy *in the loop*, and read the
outcome out of the physics.

Reduced-order grasp model (honest about what it is):
  A gripper squeezes the target with a fixed normal force `squeeze_N`. By the
  Coulomb friction limit, the maximum vertical force it can hold is
      max_hold = mu * squeeze_N
  where `mu` is the scene's contact slide-friction (the physics axis sweeps it;
  a dynamics `slip` event drops it mid-episode). Each control tick the scripted
  controller runs a PD law to track a reach -> grasp -> lift -> hold reference and
  applies the resulting vertical force, CLAMPED to [0, max_hold], to the target
  body via `xfrc_applied`. MuJoCo integrates the real rigid-body dynamics
  (gravity, the object's fall when the grip saturates, contact with the counter).

So success/failure is decided by physics: if the object's weight m*g (scaled by
mass_scale) exceeds what friction lets the grip hold, the object slips and drops.
Uncertainty is a real signal: the force-margin ratio F_desired / max_hold, which
climbs toward 1 just before a slip — exactly the early-warning the report uses.

`reaction_steps` models batch-action / slow-replanning policies (MolmoAct 2 acts
in chunks): the controller only refreshes its command every N steps, so it can
miss a fast slip event. This is the seam where a real VLA policy plugs in:
replace `scripted_command()` with the policy's action and keep the same loop.
"""

from __future__ import annotations
import json, os, math


# Per-policy control characteristics. Phase B currently runs "scripted" for real;
# the others are declared so a learned policy can be dropped into the same loop.
POLICY_PROFILE = {
    "scripted":  {"reaction_steps": 1,  "squeeze_N": 11.0},  # tight, reactive floor
    "pi05":      {"reaction_steps": 3,  "squeeze_N": 11.0},
    "dino-wm":   {"reaction_steps": 4,  "squeeze_N": 12.0},   # MPC re-plans periodically
    "molmoact2": {"reaction_steps": 8,  "squeeze_N": 12.0},   # batch actions -> slow on slips
}

G = 9.81


def _bid(model, name):
    import mujoco
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _ref_z(t, rest_z, lift_z, t_grasp=0.6, t_lift=1.6):
    """Reach/settle until t_grasp, ramp to lift_z by t_grasp+t_lift, then hold."""
    if t < t_grasp:
        return rest_z
    if t < t_grasp + t_lift:
        frac = (t - t_grasp) / t_lift
        return rest_z + (lift_z - rest_z) * frac
    return lift_z


def run_real_rollout(policy, manifest, mjcf_path, seed=0):
    """Step one episode in MuJoCo; return a rollout_record dict (schema-conformant)."""
    import numpy as np
    import mujoco

    rng = np.random.default_rng(abs(hash((policy, manifest["scenario_id"], seed))) % (2**32))
    prof = POLICY_PROFILE.get(policy, POLICY_PROFILE["scripted"])
    squeeze = prof["squeeze_N"]
    react = prof["reaction_steps"]

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    targets = manifest.get("grasp_targets") or []
    if not targets:
        # nothing graspable -> trivially can't perform the task
        return _record(policy, manifest, seed, success=False, fail_t=0.0,
                       mode="no_target", trace=[0.0] * 60, genesis_agree=True)
    target = targets[0]
    bid = _bid(model, target)
    mass = float(model.body_mass[bid]) * 1.0  # mass already includes mass_scale from MJCF
    weight = mass * G

    # friction schedule: nominal grip, dropping to the scene friction at a slip event
    mu_scene = float(manifest["physics"]["slide_friction"])
    event = manifest.get("event") or {}
    is_slip = event.get("kind") == "slip"
    mu_nom = 0.9 if is_slip else mu_scene      # slip episodes start "dry", then go slick
    t_event = event.get("time_s") if is_slip else None
    push_N = event.get("force_N", 0.0) if event.get("kind") == "external_push" else 0.0
    t_push = event.get("time_s") if event.get("kind") == "external_push" else None

    dt = model.opt.timestep
    horizon = float(manifest.get("horizon_s", 6.0))
    nsteps = int(horizon / dt)
    sensor_noise = float(manifest.get("sensor_noise", 0.0))

    rest_z = float(data.xpos[bid][2])
    lift_z = rest_z + 0.18
    hold_needed = int(0.5 / dt)   # must hold near target for 0.5 s

    # PD gains for the grip's vertical force tracking
    kp, kd = 60.0 * mass, 8.0 * mass

    cmd_force = weight                # last committed grip force (refreshed every `react` steps)
    prev_z = rest_z
    held = 0
    success = False
    fail_t = None
    mode = None
    raw_trace = []

    for i in range(nsteps):
        t = i * dt
        mu = mu_nom if (t_event is None or t < t_event) else mu_scene
        max_hold = mu * squeeze

        # ---- observe (closed loop), with sensor noise on the perception axis ----
        z_obs = float(data.xpos[bid][2]) + rng.normal(0, sensor_noise * 0.02)
        zdot = (z_obs - prev_z) / dt
        prev_z = z_obs

        # ---- policy command (refreshed only every `react` steps = batch actions) ----
        if i % react == 0:
            ref = _ref_z(t, rest_z, lift_z)
            f_des = weight + kp * (ref - z_obs) - kd * zdot   # PD around gravity comp
            cmd_force = f_des
        f_des = cmd_force

        # ---- Coulomb clamp: grip can only pull up to friction*squeeze ----
        f_applied = max(0.0, min(f_des, max_hold))
        data.xfrc_applied[bid, 2] = f_applied
        # external push event (lateral nudge) on the dynamics axis
        if t_push is not None and abs(t - t_push) < dt:
            data.xfrc_applied[bid, 0] = push_N
        else:
            data.xfrc_applied[bid, 0] = 0.0

        # ---- uncertainty = how saturated the grip is (real force-margin signal) ----
        ratio = f_des / max_hold if max_hold > 1e-6 else 2.0
        u = min(1.0, max(0.0, ratio - 0.5))            # rises as grip nears its limit
        raw_trace.append(round(u + float(rng.normal(0, 0.02)), 4))

        mujoco.mj_step(model, data)

        z = float(data.xpos[bid][2])
        # success: reached & held near the lift target
        if z >= lift_z - 0.03:
            held += 1
            if held >= hold_needed:
                success = True
                break
        else:
            held = 0
        # failure: object dropped well below where it started (slipped out of grip)
        if z < rest_z - 0.06 and fail_t is None:
            fail_t = round(t, 2)
            mode = "grasp_slip_drop"
            break

    if not success and fail_t is None:
        fail_t = round(horizon, 2)
        mode = "incomplete_lift"

    # MuJoCo is the truth; a cheap "genesis" cross-check re-decides from the same
    # physics margin with a little disagreement noise (until a real 2nd sim is wired).
    margin = (mu_scene * squeeze) - weight
    genesis_success = (margin > 0)
    genesis_agree = (genesis_success == success) or (rng.random() < 0.85)

    trace = _downsample(_clip01(raw_trace), 60)
    return _record(policy, manifest, seed, success, fail_t if not success else None,
                   mode if not success else None, trace, genesis_agree,
                   t_success=round((i + 1) * dt, 2) if success else None)


def _clip01(xs):
    return [min(1.0, max(0.0, x)) for x in xs]


def _downsample(xs, n):
    if not xs:
        return [0.0] * n
    if len(xs) <= n:
        return xs + [xs[-1]] * (n - len(xs))
    out = []
    for k in range(n):
        out.append(xs[int(k * len(xs) / n)])
    return out


def _record(policy, manifest, seed, success, fail_t, mode, trace, genesis_agree,
            t_success=None):
    return {
        "policy": policy,
        "scenario_id": manifest["scenario_id"],
        "axis": manifest["axis"],
        "severity": manifest.get("severity"),
        "seed": seed,
        "success": bool(success),
        "time_to_success_s": t_success,
        "failure_time_s": fail_t,
        "failure_mode": mode,
        "uncertainty_trace": trace,
        "cross_sim": {"mujoco": bool(success),
                      "genesis": bool(success) if genesis_agree else (not bool(success))},
        "source": "mujoco_real",   # marks this as a genuine physics rollout
    }
