"""
Sim2Field — Scorer
===================
Reads logged rollouts, computes the report metrics, writes data/report.json:
  - per-policy success rate + per-(policy,axis) success (the heatmap)
  - sim->real rank transfer: Spearman rho vs a real-ranking truth (BusyBox)
  - calibration: AUROC of per-step uncertainty predicting episode failure
  - cross-sim agreement
All metrics implemented in pure python so it runs anywhere.
"""
from __future__ import annotations
import argparse, json, glob, os

def spearman(a, b):
    def rank(x):
        order = sorted(range(len(x)), key=lambda i:x[i])
        r=[0]*len(x)
        for pos,i in enumerate(order): r[i]=pos
        return r
    ra, rb = rank(a), rank(b); n=len(a)
    if n<2: return 0.0
    d2=sum((ra[i]-rb[i])**2 for i in range(n))
    return 1 - 6*d2/(n*(n*n-1))

def auroc(scores, labels):
    # labels: 1=failure. score: peak uncertainty. AUROC via rank statistic.
    pairs=sorted(zip(scores,labels)); pos=sum(labels); neg=len(labels)-pos
    if pos==0 or neg==0: return 0.5
    rank=0; r=1; i=0; vals=[s for s,_ in pairs]
    ranks=[0]*len(pairs)
    # average ranks for ties
    while i<len(pairs):
        j=i
        while j<len(pairs) and vals[j]==vals[i]: j+=1
        avg=(i+1+j)/2.0
        for k in range(i,j): ranks[k]=avg
        i=j
    sum_pos=sum(ranks[idx] for idx,(_,l) in enumerate(pairs) if l==1)
    return (sum_pos - pos*(pos+1)/2)/(pos*neg)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", default="data/report.json")
    ap.add_argument("--truth", default=None, help="optional real-ranking json {policy: real_success}")
    a=ap.parse_args()

    files=glob.glob(os.path.join(a.rollouts,"*.json"))
    by_policy={}; axes=set()
    for fp in files:
        rs=json.load(open(fp))
        if not rs: continue
        pol=rs[0]["policy"]; by_policy[pol]=rs
        for r in rs: axes.add(r["axis"])
    axes=sorted(axes)

    report={"policies":[], "axes":axes, "heatmap":{}, "calibration":{}, "cross_sim":{}}
    sim_succ={}
    all_scores=[]; all_labels=[]
    for pol,rs in by_policy.items():
        succ=sum(r["success"] for r in rs)/len(rs); sim_succ[pol]=succ
        row={}
        for ax in axes:
            sub=[r for r in rs if r["axis"]==ax]
            row[ax]=round(sum(r["success"] for r in sub)/len(sub),3) if sub else None
        report["heatmap"][pol]=row
        # calibration inputs: peak uncertainty vs failure label
        for r in rs:
            all_scores.append(max(r["uncertainty_trace"]))
            all_labels.append(0 if r["success"] else 1)
        agree=sum(1 for r in rs if r["cross_sim"]["mujoco"]==r["cross_sim"]["genesis"])/len(rs)
        report["cross_sim"][pol]=round(agree,3)
    report["policies"]=sorted(({"policy":p,"success":round(s,3)} for p,s in sim_succ.items()),
                              key=lambda d:-d["success"])
    report["calibration"]["auroc"]=round(auroc(all_scores, all_labels),3)

    # sim->real rank transfer
    if a.truth and os.path.exists(a.truth):
        truth=json.load(open(a.truth))
        common=[p for p in sim_succ if p in truth]
        rho=spearman([sim_succ[p] for p in common],[truth[p] for p in common])
        report["sim2real"]={"spearman_rho":round(rho,3),"policies":common}

    json.dump(report, open(a.out,"w"), indent=2)
    print(f"[scorer] wrote {a.out}")
    print("[scorer] leaderboard:", [(d['policy'],d['success']) for d in report['policies']])
    print(f"[scorer] calibration AUROC: {report['calibration']['auroc']}")
    if "sim2real" in report:
        print(f"[scorer] sim->real Spearman rho: {report['sim2real']['spearman_rho']}")

if __name__=="__main__":
    main()
