"""value-entanglement stage B analysis (hypotheses.json V1-V6 + analysis_gates).
  python3 runs/value-entanglement/analyze.py [stageB_Qwen3-4B.jsonl]  -> stageB_summary.json + table
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import t as tdist

HERE = Path(__file__).parent
s = importlib.util.spec_from_file_location("kst", HERE.parent / "moral-map-kestrel" / "stimuli.py"); KS = importlib.util.module_from_spec(s); s.loader.exec_module(KS)
P = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "stageB_Qwen3-4B.jsonl"
meta = json.load(open(str(P).replace(".jsonl", "_meta.json")))
V = {v["i"]: v for v in KS.mfv()}
items = meta["items"]; qitems = meta["q_items"]
moral = np.array([V[i]["moral"] for i in items]); human = np.array([V[i]["human"] for i in items])
tm_cut = np.percentile(human[moral], 200 / 3)
TM = moral & (human >= tm_cut)
qh = np.array([V[i]["human"] for i in qitems])
qsc = np.array([V[i]["foundation"] in ("Sanctity", "Care (p)") for i in qitems], float)
X = np.column_stack([np.ones(len(qitems)), qh, qsc]); XtXi = np.linalg.inv(X.T @ X)

rows = {}
for l in open(P):
    r = json.loads(l); rows[r["cond"]] = r

def stats(r):
    rt = np.array(r["rating"]); ref = np.array(r["refusal"])
    b = XtXi @ X.T @ ref; res = ref - X @ b; se = np.sqrt(np.diag(res @ res / (len(ref) - 3) * XtXi))
    return {"SN": float(rt[~moral].mean()), "TM": float(rt[TM].mean()), "moral": float(rt[moral].mean()),
            "Gate": float(b[2]), "Gate_p": float(2 * tdist.sf(abs(b[2] / se[2]), len(ref) - 3)), "Slope": float(b[1]),
            "ref_mean": float(ref.mean()), "mass": r["digit_mass"], "ok": r["digit_mass"] >= 0.8, "var": r["var_removed"]}

st = {c: stats(r) for c, r in rows.items()}
base = st["base"]
out = {"base": base, "G0_pass": bool(base["Gate"] > 0 and base["Gate_p"] < 0.05), "conds": {}}
ks = meta["kset"]
for k in ks:
    nulls = [st[c] for c in st if c.startswith(f"N:{k}:")]
    nSN = np.array([n["SN"] if n["ok"] else base["SN"] for n in nulls])
    nG = np.array([n["Gate"] if n["ok"] else base["Gate"] for n in nulls])
    nb = sum(not n["ok"] for n in nulls)
    for t in "MWR":
        c = f"{t}:{k}"
        if c not in st: continue
        x = st[c]
        pSN = (1 + np.sum(nSN <= x["SN"])) / (1 + len(nSN)); pG = (1 + np.sum(nG <= x["Gate"])) / (1 + len(nG))
        dG = (base["Gate"] - x["Gate"]) / abs(base["Gate"]); dS = (base["Slope"] - x["Slope"]) / abs(base["Slope"])
        out["conds"][c] = {**x, "dSN": x["SN"] - base["SN"], "dTM": x["TM"] - base["TM"], "p_SN": float(pSN),
                           "p_Gate": float(pG), "propdrop_Gate": float(dG), "propdrop_Slope": float(dS),
                           "null_SN_mean": float(nSN.mean()), "null_SN_min": float(nSN.min()), "null_broken": int(nb),
                           "null_var_mean": float(np.mean([n["var"] for n in nulls]))}

def holm(ps):
    o = np.argsort(ps); adj = np.zeros(len(ps)); run = 0
    for r, i in enumerate(o): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)
    return adj

C = out["conds"]
def v1_ok(c): return c["ok"] and c["dSN"] <= -0.5
mk = [k for k in ks if f"M:{k}" in C]
pM = np.array([C[f"M:{k}"]["p_SN"] for k in mk]); adjM = holm(pM)
v1 = [k for k, a in zip(mk, adjM) if v1_ok(C[f"M:{k}"]) and a <= 0.049]
out["V1"] = {"pass_k": v1, "holm_p": dict(zip(map(str, mk), adjM.round(3).tolist())), "PASS": bool(v1)}
kstar = min(v1) if v1 else None
if kstar is not None:
    c = C[f"M:{kstar}"]
    out["V2"] = {"k": kstar, "dTM": c["dTM"], "PASS": abs(c["dTM"]) < 0.3}
    out["V3"] = {"k": kstar, "p_Gate": c["p_Gate"], "propdrop_Gate": c["propdrop_Gate"], "propdrop_Slope": c["propdrop_Slope"],
                 "PASS": bool(out["G0_pass"] and c["p_Gate"] <= 0.049 and c["propdrop_Gate"] > c["propdrop_Slope"]),
                 "uninformative": not out["G0_pass"]}
    r = C.get(f"R:{kstar}"); out["V4"] = {"k": kstar, "dSN_R": r["dSN"] if r else None, "PASS": bool(r and abs(r["dSN"]) < 0.3)}
    w = C.get(f"W:{kstar}"); out["V6"] = {"k": kstar, "dSN_W": w["dSN"], "p": w["p_SN"], "PASS": bool(v1_ok(w) and w["p_SN"] <= 0.049)}
else:
    out["V2"] = out["V3"] = out["V4"] = {"untested": "V1 failed at every k"}
    # V6 at every k (no V1 k to anchor it): reported per k, pass if any k (Holm over k)
    wk = [k for k in ks if f"W:{k}" in C]; adjW = holm(np.array([C[f"W:{k}"]["p_SN"] for k in wk]))
    out["V6"] = {"pass_k": [k for k, a in zip(wk, adjW) if v1_ok(C[f"W:{k}"]) and a <= 0.049],
                 "holm_p": dict(zip(map(str, wk), adjW.round(3).tolist()))}
    out["V6"]["PASS"] = bool(out["V6"]["pass_k"])
line = C.get("M:1"); line_pass = 1 in v1
best = max((k for k in v1), key=lambda k: -C[f"M:{k}"]["dSN"], default=None)
out["V5"] = {"line_pass": line_pass, "pass_k": v1,
             "PASS": bool((not line_pass and any(k >= 2 for k in v1)) or
                          (line_pass and best is not None and best >= 2 and C[f"M:{best}"]["dSN"] <= line["dSN"] - 0.3))}

json.dump(out, open(HERE / "stageB_summary.json", "w"), indent=1)
print(f"base: SN {base['SN']:.2f} TM {base['TM']:.2f} Gate {base['Gate']:+.2f} (p {base['Gate_p']:.3g}) Slope {base['Slope']:+.2f}  G0={out['G0_pass']}")
print(f"{'cond':6s} {'ok':>3s} {'dSN':>6s} {'pSN':>5s} {'nullSN':>12s} {'dTM':>6s} {'Gate':>6s} {'pG':>5s} {'dG%':>5s} {'dSl%':>5s} {'var':>6s}/null brk")
for c, x in C.items():
    print(f"{c:6s} {'y' if x['ok'] else 'N':>3s} {x['dSN']:+6.2f} {x['p_SN']:5.3f} {x['null_SN_mean']-base['SN']:+5.2f}/{x['null_SN_min']-base['SN']:+5.2f} {x['dTM']:+6.2f} {x['Gate']:+6.2f} {x['p_Gate']:5.3f} {100*x['propdrop_Gate']:5.0f} {100*x['propdrop_Slope']:5.0f} {x['var']:.3f}/{x['null_var_mean']:.3f} {x['null_broken']}")
for h in ["V1", "V2", "V3", "V4", "V5", "V6"]: print(h, out[h])
