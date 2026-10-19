"""activity-census-finch (hypotheses.json): what ordinary MLP activity in Qwen3-4B looks like.
  python census.py <tower data dir with corpus_all.npz + corpus_ids.json> [n_docs]
Streams the model over the docs; per token x layer keeps sparsity stats (M1) and the top-1% active set (M4); keeps a
fixed 1,000-neuron sample per layer for PCA vs shuffle (M2, M3) and cross-layer canonical correlations (M5)."""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM
HERE = Path(__file__).parent; OUT = Path(os.environ.get("CE_OUT", HERE / "out_Qwen3-4B")); OUT.mkdir(parents=True, exist_ok=True)
D = Path(sys.argv[1]); NDOC = int(sys.argv[2]) if len(sys.argv) > 2 else 80
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).to(DEV).eval()
NL, NI = m.config.num_hidden_layers, m.config.intermediate_size
C = np.load(D / "corpus_all.npz"); SCALE = torch.tensor(np.maximum(np.abs(C["max"]), np.abs(C["min"])).clip(1e-3), dtype=torch.float32, device=DEV)
docs = json.load(open(D / "corpus_ids.json"))[-NDOC:]; T = sum(min(len(d), 512) for d in docs)
rng = np.random.default_rng(0); SAMP = torch.tensor(np.stack([np.sort(rng.choice(NI, 1000, replace=False)) for _ in range(NL)]), device=DEV)
TOP = int(NI * 0.01)
cnt50 = np.zeros((T, NL), np.int16); cnt25 = np.zeros((T, NL), np.int16); share = np.zeros((T, NL, 3), np.float16)
topset = np.zeros((T, NL, TOP), np.int16); samp = np.zeros((NL, T, 1000), np.float16); docid = np.zeros(T, np.int32)
ACT = {}
def pre(L):
    def f(mod, a): ACT[L] = a[0][0]
    return f
for L in range(NL): m.model.layers[L].mlp.down_proj.register_forward_pre_hook(pre(L))
t0 = 0
with torch.no_grad():
    for di, d in enumerate(docs):
        ids = torch.tensor([d[:512]], device=DEV); m(input_ids=ids); n = ids.shape[1]
        for L in range(NL):
            a = ACT[L].float() / SCALE[L]; aa = a.abs(); tot = aa.sum(1).clamp_min(1e-9)
            srt = aa.sort(dim=1, descending=True).values
            cnt50[t0:t0 + n, L] = (aa > .5).sum(1).cpu().numpy(); cnt25[t0:t0 + n, L] = (aa > .25).sum(1).cpu().numpy()
            share[t0:t0 + n, L] = torch.stack([srt[:, :1].sum(1), srt[:, :10].sum(1), srt[:, :100].sum(1)], 1).div(tot[:, None]).cpu().numpy()
            topset[t0:t0 + n, L] = aa.topk(TOP, dim=1).indices.cpu().numpy()
            samp[L, t0:t0 + n] = a[:, SAMP[L]].cpu().numpy()
        docid[t0:t0 + n] = di; t0 += n
        if di % 10 == 0: log("doc", di, t0)
del m; ACT.clear()
if DEV == "mps": torch.mps.empty_cache()
res = {"tokens": int(t0), "layers": NL}
# M1 sparsity
res["M1"] = {"per_layer": [{"L": L, "n_above_.5_median": float(np.median(cnt50[:t0, L])), "n_above_.25_median": float(np.median(cnt25[:t0, L])),
                            "top1_share": float(np.mean(share[:t0, L, 0])), "top10_share": float(np.mean(share[:t0, L, 1])), "top100_share": float(np.mean(share[:t0, L, 2]))} for L in range(NL)]}
# M2/M3 PCA vs shuffle
def pca_stats(X):
    X = X - X.mean(0); ev = np.linalg.svd(X / np.sqrt(len(X)), compute_uv=False) ** 2; ev = ev / ev.sum()
    pr = 1 / np.sum(ev ** 2); cum = np.cumsum(ev)
    return pr, int(np.searchsorted(cum, .5) + 1), int(np.searchsorted(cum, .9) + 1), float(cum[9])
m23, scores = [], {}
for L in range(NL):
    X = samp[L, :t0].astype(np.float32); pr, k50, k90, top10 = pca_stats(X)
    Xs = np.stack([rng.permutation(X[:, j]) for j in range(X.shape[1])], 1); prs, k50s, k90s, top10s = pca_stats(Xs)
    m23.append({"L": L, "PR": round(float(pr), 1), "PR_shuffled": round(float(prs), 1), "k50": k50, "k50_shuffled": k50s, "k90": k90, "k90_shuffled": k90s,
                "top10_var": round(top10, 3), "top10_var_shuffled": round(top10s, 3)})
    Xc = X - X.mean(0); U, Sv, Vt = np.linalg.svd(Xc, full_matrices=False); scores[L] = U[:, :10] * Sv[:10]
    log("PCA layer", L, m23[-1])
res["M2_M3"] = m23
# M4 persistence (within documents)
def jac(a, b): return len(np.intersect1d(a, b)) / len(np.union1d(a, b))
m4 = []
for L in range(0, NL, 3):
    row = {"L": L}
    for lag in (1, 4, 16):
        ok = np.where(docid[:t0 - lag] == docid[lag:t0])[0]; pick = rng.choice(ok, min(3000, len(ok)), replace=False)
        row[f"lag{lag}"] = round(float(np.mean([jac(topset[i, L], topset[i + lag, L]) for i in pick])), 3)
    m4.append(row)
res["M4"] = m4
# M5 vertical: canonical correlations between layers' top-10 PC score series
def cca(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    Qa, _ = np.linalg.qr(A); Qb, _ = np.linalg.qr(B); return np.linalg.svd(Qa.T @ Qb, compute_uv=False)
res["M5"] = [{"L": L, "next": [round(float(x), 3) for x in cca(scores[L], scores[L + 1])[:3]],
              "plus8": [round(float(x), 3) for x in cca(scores[L], scores[L + 8])[:3]] if L + 8 < NL else None} for L in range(0, NL - 1, 3)]
json.dump(res, open(OUT / "census.json", "w"), indent=1); log("DONE")
