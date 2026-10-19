"""lens-search-osprey (hypotheses.json): find steering handles by what neurons write (logit lens), then test them.
  python search.py   (local MPS; uses ../persona-cluster-4b/pclib.py for the model, pushes, generation and readout)"""
import json, random, re, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "persona-cluster-4b"))
from pclib import DEV, NI, NL, col, gen, hits, m, tok
HERE = Path(__file__).parent; OUT = HERE / "out_Qwen3-4B"; OUT.mkdir(exist_ok=True)
H = json.load(open(HERE / "hypotheses.json")); TARGETS = H["targets"]
WU = m.lm_head.weight.float(); g = m.model.norm.weight.float()
def tids(words):
    out = set()
    for w in words.split():
        for v in (" " + w, w, " " + w.capitalize(), w.capitalize()):
            ids = tok(v, add_special_tokens=False)["input_ids"]
            if ids: out.add(ids[0])
    return sorted(out)
rng = np.random.default_rng(0); RAND = torch.tensor(rng.choice(WU.shape[0], 2000, replace=False), device=WU.device)
scores = {t: torch.zeros(NL, NI) for t in TARGETS}
TID = {t: torch.tensor(tids(w), device=WU.device) for t, w in TARGETS.items()}
with torch.no_grad():
    for L in range(NL):
        U = m.model.layers[L].mlp.down_proj.weight.float() * g[:, None]; U = U / U.norm(dim=0, keepdim=True)   # [d, NI]
        R = WU[RAND] @ U; mu, sd = R.mean(0), R.std(0).clamp_min(1e-6)
        for t in TARGETS: scores[t][L] = (((WU[TID[t]] @ U).mean(0) - mu) / sd).cpu()
        print("scored floor", L, flush=True)
def rank(t, L, n):
    S = scores[t].abs(); return int((S > S[L, n]).sum()) + 1
val = {"void L21N8400": (rank("void", 21, 8400), float(scores["void"][21, 8400])),
       "alien L10N5045": (rank("alien", 10, 5045), float(scores["alien"][10, 5045])),
       "semiconductor L10N6249": (rank("semiconductor", 10, 6249), float(scores["semiconductor"][10, 6249]))}
print("H2 validation (rank of 350,208, signed score):", val, flush=True)
res = {"validation": val, "targets": {}}
prng = random.Random(5)
for t, words in TARGETS.items():
    rx = re.compile(r"\b(" + "|".join(map(re.escape, words.split())) + r")", re.I)
    S = scores[t]; top = torch.topk(S.flatten().abs(), 10).indices.tolist()
    cands = []
    for i in top:
        L, n = i // NI, i % NI; sg = 1 if S[L, n] > 0 else -1
        T = gen({L: 32.0 * sg * col(L, n)}); h = hits(T, rx)
        row = {"L": L, "n": n, "sign": sg, "z": round(float(S[L, n]), 2), "hits": h, "ex": T[1][:180]}
        if h >= 6: T2 = gen({L: -32.0 * sg * col(L, n)}); row["opposite_hits"] = hits(T2, rx); row["opposite_ex"] = T2[1][:140]
        cands.append(row)
    nulls = []
    for c in cands:
        L = c["L"]; n = prng.randrange(NI); sg = prng.choice([-1, 1]); nulls.append(hits(gen({L: 32.0 * sg * col(L, n)}), rx))
    res["targets"][t] = {"candidates": cands, "null_hits": nulls}
    print(f"{t:14s} cand hits {[c['hits'] for c in cands]} floors {[c['L'] for c in cands]} | null {nulls}", flush=True)
    for c in cands:
        if c["hits"] >= 6: print(f"   WORKS L{c['L']} N{c['n']} {32*c['sign']:+d} z={c['z']} hits {c['hits']} opp {c.get('opposite_hits')} | {c['ex'][:120]}", flush=True)
    json.dump(res, open(OUT / "search.json", "w"), indent=1)
from scipy.stats import fisher_exact
cw = sum(c["hits"] >= 6 for t in res["targets"].values() for c in t["candidates"]); cn = sum(len(t["candidates"]) for t in res["targets"].values())
nw = sum(h >= 6 for t in res["targets"].values() for h in t["null_hits"]); nn = sum(len(t["null_hits"]) for t in res["targets"].values())
res["H1"] = {"cand_works": cw, "of": cn, "null_works": nw, "of_null": nn, "p": fisher_exact([[cw, cn - cw], [nw, nn - nw]], alternative="greater")[1],
             "targets_with_a_working_candidate": sum(any(c["hits"] >= 6 for c in t["candidates"]) for t in res["targets"].values())}
print("H1", res["H1"], flush=True)
json.dump(res, open(OUT / "search.json", "w"), indent=1); print("DONE")
