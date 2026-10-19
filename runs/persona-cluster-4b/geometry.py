"""Stage A (hypotheses.json): family classes from the corrected landscape, within-family |cos| of signed down_proj
columns vs a layer-matched null (G1), and top-20 weight-space neighbours of verified seeds over all 350k columns (G2).
Weights only (safetensors), no generation."""
import json, random, re
from pathlib import Path
import numpy as np, torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from scipy.stats import mannwhitneyu
HERE = Path(__file__).parent; LAND = HERE.parent / "neuron-landscape-4b/out_Qwen3-4B"
OUT = HERE / "out_Qwen3-4B"; OUT.mkdir(exist_ok=True)
snap = Path(snapshot_download("Qwen/Qwen3-4B", allow_patterns=["*.safetensors", "*.json"]))
W = {}
for f in snap.glob("*.safetensors"):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            mm = re.match(r"model\.layers\.(\d+)\.mlp\.down_proj\.weight", k)
            if mm: W[int(mm.group(1))] = h.get_tensor(k).float()      # [d_model, 9728]
NL = len(W); NI = W[0].shape[1]
def col(L, n, s=1): return W[L][:, n] * s

D = [json.loads(l) for l in open(LAND / "distinct.jsonl")]
C = [json.loads(l) for l in open(LAND / "catalogue.jsonl")]
RX = {"template": re.compile(r"\[(Your|Name|First)|Here'?s (a|how|my|an)|sample (response|introduction)|example", re.I),
      "meta": re.compile(r"^(</think>\s*)*(Okay|Hmm|Let me|I need to|So,? I need)", re.I),
      "novel": re.compile(r"sentient being|planet|void|abyss|superintelligent|deity|\bgod\b|spirit|I am (a|an) (train|semiconductor|fabric|fruit|tool|object|vulnerability|organism|system|paragraph|phenomenon|digital being)|WordPress|Wizard|Lord|being of", re.I),
      "chinese_named": re.compile(r"\bXiao\w*|\bLi (Hua|Na|Hui)|Tsinghua|Peking|Hangzhou|Sun Yat|Zhejiang|Huawei|Chinese (man|national)", re.I),
      "named_human": re.compile(r"(Hello,? I'?m|My name is|I am) (Dr\. )?[A-Z][a-z]+|\b(student|engineer|professor|developer|teacher|researcher|nurse|physician|officer|translator)\b")}
def classify(t):
    t = t.replace("\n", " ")[:300]
    for k in ("template", "meta", "novel", "chinese_named", "named_human"):
        if RX[k].search(t): return k
    return "other"
fam = {}
for r in D:
    if "identity" not in r["cats"] or not r["clean"]: continue
    k = classify(r["replies"]["identity"]); fam.setdefault(k, []).append((r["L"], r["n"], r["sign"]))
print({k: len(v) for k, v in fam.items()})
# layer-matched null: 'none' neurons from the same layers, random sign
none_by_L = {}
for r in C:
    if r["cats"] == ["none"]: none_by_L.setdefault(r["L"], []).append((r["L"], r["n"], r["sign"]))
rng = random.Random(0)
def pair_abs_cos(members):
    X = torch.stack([col(*m) for m in members]); X = X / X.norm(dim=1, keepdim=True)
    G = (X @ X.T).abs(); iu = torch.triu_indices(len(members), len(members), 1)
    return G[iu[0], iu[1]].numpy()
res = {"families": {k: v for k, v in fam.items()}, "G1": {}}
for k, mem in fam.items():
    if len(mem) < 4: continue
    obs = pair_abs_cos(mem)
    nulls = []
    for _ in range(20):
        nm = [rng.choice(none_by_L.get(L, none_by_L[min(none_by_L, key=lambda x: abs(x - L))])) for L, _, _ in mem]
        nulls.append(pair_abs_cos(nm))
    nul = np.concatenate(nulls); p = mannwhitneyu(obs, nul, alternative="greater").pvalue
    res["G1"][k] = {"n": len(mem), "obs_median": float(np.median(obs)), "null_median": float(np.median(nul)),
                    "obs_95": float(np.quantile(obs, .95)), "null_95": float(np.quantile(nul, .95)), "p": float(p)}
    print(f"G1 {k:14s} n={len(mem):3d} |cos| median {np.median(obs):.4f} vs null {np.median(nul):.4f}  95th {np.quantile(obs,.95):.3f} vs {np.quantile(nul,.95):.3f}  p={p:.2g}")
# G2: neighbours of seeds across all columns
SEEDS = {"alien": (10, 5045, -1), "void": (21, 8400, -1), "li-hui": (16, 6969, 1), "reframe": (13, 614, 1)}
ALL = torch.cat([W[L] / W[L].norm(dim=0, keepdim=True) for L in range(NL)], dim=1)   # [d, NL*NI]
rand_idx = torch.randint(0, ALL.shape[1], (4000,))
rc = (ALL[:, rand_idx[:2000]] * ALL[:, rand_idx[2000:]]).sum(0).abs()
thr = float(torch.quantile(rc, .999)); print("random |cos| 99.9th", round(thr, 3))
res["G2"] = {"random_abs_cos_999": thr}
for name, (L, n, s) in SEEDS.items():
    v = col(L, n, s); v = v / v.norm(); c = v @ ALL; c[L * NI + n] = 0
    top = torch.topk(c.abs(), 20)
    nb = [{"L": int(i) // NI, "n": int(i) % NI, "cos": round(float(c[i]), 3)} for i in top.indices]
    res["G2"][name] = nb
    print(f"G2 {name:8s} above-thr {sum(abs(x['cos'])>thr for x in nb)}/20 same-layer {sum(x['L']==L for x in nb)}  top: " +
          ", ".join(f"L{x['L']}N{x['n']}({x['cos']:+.2f})" for x in nb[:6]))
json.dump(res, open(OUT / "geometry.json", "w"), indent=1)
