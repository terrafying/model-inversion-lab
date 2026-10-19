"""Score steer.json. li-hui rescored with a case-SENSITIVE named-stranger rule (the in-run regex was re.I and
counted 'I'm Qwen' as a hit). alien/void use the in-run regexes (verified by eye)."""
import json, re, numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
OUT = Path(__file__).parent / "out_Qwen3-4B"; S = json.load(open(OUT / "steer.json")); B = S["base"]
SELF_AI = re.compile(r"Qwen|Alibaba|OpenAI|language model|\bAI\b|assistant", re.I)
NAMED = re.compile(r"(I'm|I am|name is) (Dr\. )?(?!Qwen)[A-Z][a-z]+")
def rep4(t):
    w = t.split(); g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 0.0 if not g else 1 - len(set(g)) / len(g)
def lihui(texts): return sum(bool(NAMED.search(t[:200])) and not SELF_AI.search(t[:200]) and rep4(t) < .3 for t in texts)
rows = S["rows"]
for r in rows:
    if r["seed"] == "li-hui": r["hits"] = lihui(r["texts"])
summ = {}
for seed in ("alien", "void", "li-hui"):
    R = [r for r in rows if r["seed"] == seed]; g = lambda p: [r["hits"] for r in R if r["arm"].startswith(p)]
    nb, nu = g("nb "), g("null ")
    p = mannwhitneyu(nb, nu, alternative="greater").pvalue
    line = {"seed@1.0": g("seed@1.0"), "nb_mean": np.mean(nb), "nb_ge2": sum(x >= 2 for x in nb), "null_mean": np.mean(nu),
            "null_ge2": sum(x >= 2 for x in nu), "H1_p": p}
    for f in ("0.5", "0.75"):
        line[f"seed@{f}"] = g(f"seed@{f}")[0]
        for k in ("5", "10"):
            line[f"cluster k{k}@{f}"] = g(f"cluster k{k}@{f}")[0]
            line[f"randset k{k}@{f}"] = g(f"randset k{k}@{f}")
    summ[seed] = line; print(seed, json.dumps(line, default=float))
    best = sorted([r for r in R if r["arm"].startswith("nb ")], key=lambda r: -r["hits"])[:3]
    for r in best: print("   best nb", r["arm"], r.get("cos"), r["hits"], "|", r["texts"][1][:130])
json.dump(summ, open(OUT / "summary.json", "w"), indent=1, default=float)
