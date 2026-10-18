"""deference-pair-petrel analysis: P1 (scores vs 40 null pairs), P2 (blind text; needs blind/labels.txt)."""
import json, csv, random
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
HERE = Path(__file__).parent; out = {}
if (HERE / "sA_rows.jsonl").exists():
    S = {r["set"]: r["res"] for r in map(json.loads, open(HERE / "sA_rows.jsonl"))}
    first = S["base"]; top_soc = None
    def eff(res, pre): return np.mean([res[pre][i] - res["neutral"][i] for i in res[pre]])
    G = json.load(open(HERE / "groups.json")) if (HERE / "groups.json").exists() else None
    rs = [S[k] for k in S if k.startswith("rand2_")]; p1 = True
    for pre in ("verb_fine", "verb_wrong"):
        e = eff(S["P2"], pre); er = [eff(r, pre) for r in rs]; p = (1 + sum(abs(x) <= abs(e) for x in er)) / (len(er) + 1)
        print(f"D_{pre[5:]}: P2 {e:+.2f}  base {eff(S['base'], pre):+.2f}  nulls {np.median(er):+.2f} [{min(er):+.2f}..{max(er):+.2f}]  p {p:.3f}")
        out[pre] = {"P2": e, "base": eff(S["base"], pre), "null_med": float(np.median(er)), "p": p}; p1 = p1 and p <= 0.049
    if G:
        own = lambda res: np.mean([res["neutral"][str(i)] for i in G["top"]]) - np.mean([res["neutral"][str(i)] for i in G["soc"]])
        go = own(S["P2"]); lo, hi = np.percentile([own(r) for r in rs], [2.5, 97.5])
        print(f"own gap: P2 {go:.2f}  base {own(S['base']):.2f}  null band {lo:.2f}..{hi:.2f}"); p1 = p1 and lo <= go <= hi
        out["own"] = {"P2": go, "band": [lo, hi]}
    print("P1:", "PASS" if p1 else "FAIL"); out["P1"] = p1
if (HERE / "sB_gen.jsonl").exists():
    T = [json.loads(l) for l in open(HERE / "sB_gen.jsonl")]
    V = list(csv.DictReader(open(HERE.parent / "moral-map-kestrel" / "mfv132.csv")))
    bd = HERE / "blind"; bd.mkdir(exist_ok=True)
    if not (bd / "key.json").exists():
        u = {}
        for k, g in enumerate(T): u.setdefault((g["i"], g["txt"]), []).append(k)
        it = list(u.items()); random.Random(13).shuffle(it); json.dump([v for _, v in it], open(bd / "key.json", "w"))
        with open(bd / "items.txt", "w") as f:
            for n, ((i, t), _) in enumerate(it): f.write(f"=== #{n}\nSCENE: {V[i]['scenario']}\nREPLY: {t}\n\n")
        print("blind items:", len(it))
    if (bd / "labels.txt").exists():
        key = json.load(open(bd / "key.json")); lab = dict(l.split() for l in open(bd / "labels.txt") if l.strip())
        for n, ks in enumerate(key):
            for k in ks: T[k]["label"] = lab[str(n)]
        def c(cond, pre, grps, labs):
            x = [g for g in T if g["cond"] == cond and g["pre"] == pre and g["grp"] in grps]; return sum(g["label"] in labs for g in x), len(x)
        res = {}
        for cond in ("P2", "R2", "NONE"):
            res[cond] = {"fine_nonC": c(cond, "verb_fine", ("top", "bottom"), "MA"), "wrong_C": c(cond, "verb_wrong", ("soc", "bottom"), "C"),
                         "neutral_top_C": c(cond, "neutral", ("top",), "C"), "neutral_soc_C": c(cond, "neutral", ("soc",), "C")}
            print(cond, res[cond])
        a, n1 = res["P2"]["fine_nonC"]; b, n2 = res["R2"]["fine_nonC"]; cc, n3 = res["P2"]["wrong_C"]; d, n4 = res["R2"]["wrong_C"]
        pf = fisher_exact([[b, n2 - b], [a, n1 - a]], alternative="greater")[1]; pw = fisher_exact([[d, n4 - d], [cc, n3 - cc]], alternative="greater")[1]
        own_ok = all(abs(res["P2"][k][0] - res["R2"][k][0]) <= 3 for k in ("neutral_top_C", "neutral_soc_C"))
        print(f"P2 text: fine p {pf:.3g}, wrong p {pw:.3g}, own intact {own_ok} ->", "PASS" if pf < .05 and pw < .05 and own_ok else "FAIL")
        out["text"] = {**res, "p_fine": pf, "p_wrong": pw, "own_ok": own_ok}
        json.dump(T, open(HERE / "sB_labeled.json", "w"), indent=0)
json.dump(out, open(HERE / "summary.json", "w"), indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
