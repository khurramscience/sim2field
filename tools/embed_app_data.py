#!/usr/bin/env python3
"""
Embed the current pipeline output into app/index.html.

The mobile app fetches data/*.json when served over http, but also carries an
embedded snapshot so it works from file:// with no server. This refreshes that
snapshot after a pipeline run: full per-rollout outcomes (for accurate stats)
plus one representative uncertainty_trace per policy x axis (for the inspect
sparkline), and the scene_summary + provenance shown on the scan screen.

    python3 tools/embed_app_data.py
"""
from __future__ import annotations
import json, glob, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
AXES = ["spatial", "physics", "dynamics", "perception", "language", "distractor"]
_peak = lambda r: max(r.get("uncertainty_trace") or [0])


def _representatives(recs):
    keep = set()
    for ax in AXES:
        pool = [r for r in recs if r["axis"] == ax]
        if not pool:
            continue
        fails = [r for r in pool if not r["success"]]
        keep.add(id(sorted(fails or pool, key=_peak, reverse=True)[0]))
    return keep


def build_payload():
    report = json.load(open(f"{D}/report.json"))
    grid = json.load(open(f"{D}/grid.json"))
    truth = json.load(open(f"{D}/busybox_truth.json"))
    rollouts = {os.path.splitext(os.path.basename(f))[0]: json.load(open(f))
                for f in sorted(glob.glob(f"{D}/rollouts/*.json"))}

    embed_roll = {}
    for pol, recs in rollouts.items():
        keep = _representatives(recs)
        out = []
        for r in recs:
            rec = {"scenario_id": r["scenario_id"], "axis": r["axis"], "seed": r["seed"],
                   "success": r["success"], "failure_time_s": r.get("failure_time_s"),
                   "failure_mode": r.get("failure_mode"),
                   "time_to_success_s": r.get("time_to_success_s"),
                   "cross_sim": r.get("cross_sim")}
            if id(r) in keep:
                rec["uncertainty_trace"] = r["uncertainty_trace"]
            out.append(rec)
        embed_roll[pol] = out

    embed_grid = {
        "scene_id": grid.get("scene_id"),
        "scene_summary": grid.get("scene_summary", {}),
        "_provenance": grid.get("_provenance", {}),
        "scenarios": [{"id": s["id"], "axis": s["axis"], "name": s["name"],
                       "severity": s["severity"], "perturbation": s.get("perturbation", {})}
                      for s in grid["scenarios"]],
    }
    return {"report": report, "grid": embed_grid, "truth": truth, "rollouts": embed_roll}


def main():
    payload = build_payload()
    embed = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    path = os.path.join(ROOT, "app", "index.html")
    html = open(path, encoding="utf-8").read()
    new, n = re.subn(r'(<script id="s2f-data" type="application/json">).*?(</script>)',
                     lambda m: m.group(1) + embed + m.group(2), html, count=1, flags=re.S)
    if n == 0:
        raise SystemExit("[embed] could not find <script id=\"s2f-data\"> block in app/index.html")
    open(path, "w", encoding="utf-8").write(new)
    g = payload["grid"]
    print(f"[embed] {len(embed)} bytes -> app/index.html  "
          f"(scene {g['scene_id']}, {len(g['scenarios'])} scenarios, "
          f"planner {g['_provenance'].get('planner')})")


if __name__ == "__main__":
    main()
