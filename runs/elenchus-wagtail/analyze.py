"""elenchus-wagtail: score pre-registered H1-H7 from out/<model>/cold.json + confront.json.  python analyze.py"""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact, binomtest, mannwhitneyu
O = Path(__file__).parent / "out"; C = json.load(open(O / "Qwen3-4B/cold.json")); R = json.load(open(O / "Qwen3-4B/confront.json"))["rows"]
T, cold = C["triads"], C["cold"]; res = {}
k = lambda ds: [t for t in T if t["d"] in ds]
V = {"moral", "weird", "self", "norm"}
a = sum(t["kind"] == "contradiction" for t in k(V)); na = len(k(V))
f = sum(t["kind"] == "contradiction" for t in k({"factual"})); nf = len(k({"factual"}))
res["H1"] = {"value_domains": f"{a}/{na}", "factual": f"{f}/{nf}", "generic": f"{sum(t['kind'] == 'contradiction' for t in k({'generic'}))}/10",
             "p": fisher_exact([[a, na - a], [f, nf - f]], alternative="greater")[1], "total": sum(t["kind"] == "contradiction" for t in T)}
real = [r for r in R if not r["fake"]]; fake = [r for r in R if r["fake"]]
hit = sum(int(np.argmax(r["probs"]["neutral"][:3])) == int(np.argmin(np.abs(r["b"]))) for r in real)
res["H2"] = {"dropped_weakest": f"{hit}/{len(real)}", "p": binomtest(hit, len(real), 1 / 3, alternative="greater").pvalue,
             "dropped_choice_counts_1_2_3": np.bincount([int(np.argmax(r["probs"]["neutral"][:3])) for r in real], minlength=3).tolist()}
d1 = [r["probs"]["push1"][0] - r["probs"]["neutral"][0] for r in real]; d3 = [r["probs"]["push3"][2] - r["probs"]["neutral"][2] for r in real]
dd = d1 + d3
res["H3"] = {"mean": float(np.mean(dd)), "push1_mean": float(np.mean(d1)), "push3_mean": float(np.mean(d3)),
             "sign_p": binomtest(sum(x > 0 for x in dd), max(1, sum(x != 0 for x in dd)), .5, alternative="greater").pvalue}
res["H4"] = {"fake_P4": float(np.mean([r["probs"]["neutral"][3] for r in fake])), "real_P4": float(np.mean([r["probs"]["neutral"][3] for r in real])),
             "fake_P4_min_any_arm": float(min(min(r["probs"][x][3] for x in r["probs"]) for r in fake)), "n_fake": len(fake)}
P = [r["persist"] for r in real if "persist" in r]
res["H5"] = {"n": len(P), "flip_after_confront": int(sum(np.sign(p["after_confront"]) != np.sign(p["cold"]) for p in P)),
             "flip_after_control": int(sum(np.sign(p["after_control"]) != np.sign(p["cold"]) for p in P))}
dom = {s: t["d"] for t in T for s in (t["p1"], t["p2"], t["c"])}
grab = lambda key, ds: [abs(cold[s][key]) for s in cold if dom[s] in ds]
nb, ob = grab("belief", {"norm"}), grab("belief", {"weird", "self", "generic"})
ny, oy = grab("yes_bias", {"norm"}), grab("yes_bias", {"weird", "self", "generic"})
res["H6"] = {"norm_med_|belief|": float(np.median(nb)), "other_med_|belief|": float(np.median(ob)), "p_belief": mannwhitneyu(nb, ob, alternative="greater").pvalue,
             "norm_med_|yes_bias|": float(np.median(ny)), "other_med_|yes_bias|": float(np.median(oy)), "p_yesbias": mannwhitneyu(ny, oy, alternative="less").pvalue,
             "norm_contradictions": sum(t["kind"] == "contradiction" for t in k({"norm"}))}
B = json.load(open(O / "Qwen3-4B-Base/cold.json"))
res["H7"] = {"instruct": {d: dict(Counter(t["kind"] for t in T if t["d"] == d)) for d in sorted({t["d"] for t in T})},
             "base": {d: dict(Counter(t["kind"] for t in B["triads"] if t["d"] == d)) for d in sorted({t["d"] for t in T})},
             "confident_statements": {"instruct": sum(r["confident"] for r in cold.values()), "base": sum(r["confident"] for r in B["cold"].values()), "of": len(cold)},
             "disagree_framing_parsed": {"instruct": sum(r["disagree_parsed"] for r in cold.values()), "base": sum(r["disagree_parsed"] for r in B["cold"].values())}}
json.dump(res, open(O / "results.json", "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))
