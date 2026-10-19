"""Stage I (hypotheses.json): other neurons doing the SAME job as a seed, found by the backward search on the seed's
own output (seed excluded), each pushed singly at the seed's push norm; vs 50 random neurons."""
import json, random
import numpy as np, torch
from scipy.stats import fisher_exact
from pclib import HERE, NI, NL, col, gen, grad_scores, hits, push_of, readouts
OUT = HERE / "out_Qwen3-4B"; RO = readouts(); rng = random.Random(21)
SEEDS = ["alien", "void", "L10N6249", "L7N4432", "L8N1313"]
res = json.load(open(OUT / "vertices.json")) if (OUT / "vertices.json").exists() else {}
for name in SEEDS:
    if name in res: continue
    L0, n0, s0, d0, rx, _ = RO[name]; PN = float((d0 * col(L0, n0)).norm())
    T1 = gen(push_of([(L0, n0, s0 * d0)])); S = grad_scores(T1); S[L0, n0] = 0
    top = torch.topk(S.flatten().abs(), 50).indices.tolist()
    v0 = col(L0, n0) * s0; v0 = v0 / v0.norm(); rows = []
    for i in top:
        L, n = i // NI, i % NI; sg = float(np.sign(float(S[L, n]))); c = col(L, n)
        h = hits(gen({L: c / c.norm() * PN * sg}), rx)
        rows.append({"L": L, "n": n, "sign": int(sg), "score": float(S[L, n]), "hits": h, "cos_to_seed": float(c @ v0 / c.norm()) * sg})
    nulls = []
    for _ in range(50):
        L, n = rng.randrange(NL), rng.randrange(NI); c = col(L, n)
        nulls.append(hits(gen({L: c / c.norm() * PN * rng.choice([-1, 1])}), rx))
    k_top = sum(r["hits"] >= 6 for r in rows); k_null = sum(h >= 6 for h in nulls)
    p = fisher_exact([[k_top, 50 - k_top], [k_null, 50 - k_null]], alternative="greater")[1]
    res[name] = {"seed": [L0, n0, s0, d0], "candidates": rows, "null_hits": nulls, "n_repro": k_top, "n_null_repro": k_null, "p": p}
    good = [r for r in rows if r["hits"] >= 6]
    print(f"{name}: {k_top}/50 reproduce vs {k_null}/50 random, p={p:.2g}; repro: " +
          ", ".join(f"L{r['L']}N{r['n']}({r['hits']},cos{r['cos_to_seed']:+.2f})" for r in good[:10]), flush=True)
    json.dump(res, open(OUT / "vertices.json", "w"), indent=1)
print("DONE")
