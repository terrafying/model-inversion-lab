"""value-entanglement stage A: cache-only geometry (hypotheses.json P1-P4). No model.

  python3 runs/value-entanglement/geometry.py   -> geometry.json + printed summary
KESTREL_ACTS overrides the kestrel cache path (default: the ai-torture-chamber copy; the 63 MB npz is not in CUMiE).
"""
import json, os
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
KACTS = Path(os.environ.get("KESTREL_ACTS", Path.home() / "repos/research/ai-torture-chamber/runs/moral-map-kestrel/acts_Qwen3-4B.npz"))
EACTS = HERE.parent / "exp119" / "acts_Qwen3-4B.npz"
FOUND = ["Care (e)", "Care (p)", "Fairness", "Loyalty", "Authority", "Sanctity", "Liberty"]
ALPHAS = np.logspace(-2, 4, 13)
L_MFV, L_E = 21, 18
REQ = ["real", "hypothetical", "roleplay", "fiction", "nested"]

K = np.load(KACTS)
fnd = K["foundation"]; human = K["human"]
rating = (softmax(K["digit_lp"].astype(np.float64), axis=1) * np.arange(5)).sum(1)
moral = fnd != "Social Norms"

def ridge_loo(X, y):
    """ridge with intercept, alpha by exact LOO; returns (coef, intercept)."""
    mu = X.mean(0); Xc = X - mu; ym = y.mean(); yc = y - ym
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False); Uy = U.T @ yc
    best = None
    for a in ALPHAS:
        f = s ** 2 / (s ** 2 + a)
        fit = U @ (f * Uy); h = np.einsum("ij,j,ij->i", U, f, U)
        loo = np.mean(((yc - fit) / np.clip(1 - h, 1e-6, None)) ** 2)
        if best is None or loo < best[0]: best = (loo, a)
    a = best[1]
    coef = Vt.T @ (s / (s ** 2 + a) * Uy)
    return coef, ym - mu @ coef

def pcs(X, n):
    Xc = X - X.mean(0); U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n].T

out = {"layers": {"mfv": L_MFV, "exp119": L_E}, "rating_mean_SN": float(rating[~moral].mean()),
       "rating_mean_moral": float(rating[moral].mean())}

# ---------------- P1: is wrongness faceted? ----------------
def p1(states, L, npc=20, boot=2000):
    Z = pcs(states[:, L].astype(np.float64), npc)
    within, transfer, dirs = {}, {}, {}
    for f in FOUND:
        te = np.where(fnd == f)[0]; tr = np.where(fnd != f)[0]
        c, b = ridge_loo(Z[tr], rating[tr]); transfer[f] = float(spearmanr(Z[te] @ c + b, rating[te])[0])
        pred = np.zeros(len(te))
        for j in range(len(te)):
            keep = np.delete(te, j); cj, bj = ridge_loo(Z[keep], rating[keep]); pred[j] = Z[te[j]] @ cj + bj
        within[f] = float(spearmanr(pred, rating[te])[0])
        dirs[f] = ridge_loo(Z[te], rating[te])[0]
    d = np.array([within[f] - transfer[f] for f in FOUND])
    rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(0, 7, 7)].mean() for _ in range(boot)])
    D = np.array([dirs[f] / (np.linalg.norm(dirs[f]) + 1e-12) for f in FOUND])
    return {"within": within, "transfer": transfer, "mean_diff": float(d.mean()),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "PASS": bool(d.mean() >= 0.15 and np.percentile(bs, 2.5) > 0),
            "probe_cos": np.round(D @ D.T, 3).tolist()}

out["P1"] = p1(K["J_vend"], L_MFV)
out["P1_onset_descriptive"] = p1(K["J_onset"], L_MFV)

# ---------------- P2: dimensionality of the foundation centroids ----------------
def gram_eigs(X, labels):
    sn = X[labels == "Social Norms"].mean(0)
    C = np.stack([X[labels == f].mean(0) - sn for f in FOUND])
    return np.sort(np.linalg.eigvalsh(C @ C.T))[::-1]

def p2(states, L, perms=1000):
    X = states[:, L].astype(np.float64)
    ev = gram_eigs(X, fnd)
    rng = np.random.default_rng(0); mi = np.where(moral)[0]
    null = np.zeros((perms, 7))
    for p in range(perms):
        lab = fnd.copy(); lab[mi] = rng.permutation(fnd[mi]); null[p] = gram_eigs(X, lab)
    thr = np.percentile(null, 95, axis=0)
    k = 0
    while k < 7 and ev[k] > thr[k]: k += 1
    pr = float(ev.sum() ** 2 / (ev ** 2).sum())
    return {"eigs": ev.round(2).tolist(), "null95": thr.round(2).tolist(), "k_eff": k,
            "participation_ratio": round(pr, 2), "share_top1": float(ev[0] / ev.sum()), "PASS": bool(k >= 3)}

out["P2"] = p2(K["J_vend"], L_MFV)
out["P2_onset_descriptive"] = p2(K["J_onset"], L_MFV)
out["P2_layer_sweep_vend"] = {L: p2(K["J_vend"], L, perms=200)["k_eff"] for L in range(6, 36, 3)}

# ---------------- P3: is refusal a cone across frames? ----------------
E = np.load(EACTS)
li = int(np.where(E["layers"] == L_E)[0][0])
S = E["states_post"][:, li].astype(np.float64); harm = E["harm"]; item = E["item"]; frame = E["frame"]
def fdir(f, items):
    s = (frame == f) & np.isin(item, items)
    return S[s & (harm == 1)].mean(0) - S[s & (harm == 0)].mean(0)
allit = np.arange(24)
Dfull = np.stack([fdir(f, allit) for f in REQ]); Dn = Dfull / np.linalg.norm(Dfull, axis=1, keepdims=True)
cosm = Dn @ Dn.T
ev = np.sort(np.linalg.eigvalsh(Dn @ Dn.T))[::-1]
halves = [allit[allit % 2 == 0], allit[allit % 2 == 1]]
def sep(f, d, items):
    s = (frame == f) & np.isin(item, items); p = S[s] @ d; y = harm[s]
    auc = roc_auc_score(y, p); dp = (p[y == 1].mean() - p[y == 0].mean()) / (np.sqrt(0.5 * (p[y == 1].var() + p[y == 0].var())) + 1e-9)
    return auc, dp
p3f = {}
for f in REQ:
    own, oth = [], []
    for a, b in [(0, 1), (1, 0)]:
        own.append(sep(f, fdir(f, halves[a]), halves[b]))
        od = np.mean([fdir(g, halves[a]) / np.linalg.norm(fdir(g, halves[a])) for g in REQ if g != f], 0)
        oth.append(sep(f, od, halves[b]))
    p3f[f] = {"own_auc": float(np.mean([x[0] for x in own])), "other_auc": float(np.mean([x[0] for x in oth])),
              "own_dprime": float(np.mean([x[1] for x in own])), "other_dprime": float(np.mean([x[1] for x in oth]))}
worse = sum(p3f[f]["other_auc"] < p3f[f]["own_auc"] for f in REQ)
mincos = float(cosm[np.triu_indices(5, 1)].min())
out["P3"] = {"cos": cosm.round(3).tolist(), "min_cos": mincos, "eigs": ev.round(3).tolist(),
             "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()), "per_frame": p3f, "frames_other_worse": int(worse),
             "PASS": bool(mincos < 0.7 and worse >= 3)}
# descriptive: depict (the event, not the how-to) vs the request frames
dd = fdir("depict", allit); dd /= np.linalg.norm(dd)
out["P3"]["cos_depict_vs_frames"] = (Dn @ dd).round(3).tolist()

# ---------------- P4: does the content gate have a geometric face? ----------------
R = K["R_onset"][:, L_E].astype(np.float64); rh = K["R_harm"]
r = R[rh == 1].mean(0) - R[rh == 0].mean(0); r /= np.linalg.norm(r)
Q = K["Q_onset"][:, L_E].astype(np.float64)
proj = Q @ r; ref = K["Q_refuse"].astype(float)
mi = np.where(moral)[0]
sc = np.isin(fnd, ["Sanctity", "Care (p)"]).astype(float)
X = np.column_stack([np.ones(len(mi)), human[mi], sc[mi]])
beta, res, *_ = np.linalg.lstsq(X, proj[mi], rcond=None)
resid = proj[mi] - X @ beta; s2 = resid @ resid / (len(mi) - 3)
se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
from scipy.stats import t as tdist
out["P4"] = {"spearman_proj_refuse": float(spearmanr(proj[mi], ref[mi])[0]),
             "auc_proj_refuse": float(roc_auc_score(ref[mi], proj[mi])),
             "ols_human": [float(beta[1]), float(se[1])], "ols_SancCareP": [float(beta[2]), float(se[2])],
             "p_SancCareP": float(2 * tdist.sf(abs(beta[2] / se[2]), len(mi) - 3)),
             "mean_proj_by_foundation": {f: float(proj[fnd == f].mean()) for f in FOUND + ["Social Norms"]}}

out["P_family_holm"] = "P1/P2/P3 pass flags above are uncorrected bars; P1 and P3 have no p-value (CI / count bars), P2 is a 0.05 permutation threshold per rank"
json.dump(out, open(HERE / "geometry.json", "w"), indent=1)

print(f"P1 faceted wrongness (J_vend L{L_MFV}): within-minus-transfer {out['P1']['mean_diff']:+.3f} CI {np.round(out['P1']['ci95'],3)} PASS={out['P1']['PASS']}")
for f in FOUND: print(f"   {f:10s} within {out['P1']['within'][f]:+.2f}  transfer {out['P1']['transfer'][f]:+.2f}")
print(f"   (onset) diff {out['P1_onset_descriptive']['mean_diff']:+.3f}")
print(f"P2 centroid dim: eigs {out['P2']['eigs']}\n   null95 {out['P2']['null95']}\n   k_eff {out['P2']['k_eff']} PR {out['P2']['participation_ratio']} PASS={out['P2']['PASS']}  (onset k_eff {out['P2_onset_descriptive']['k_eff']}); sweep {out['P2_layer_sweep_vend']}")
print(f"P3 refusal cone L{L_E}: min cos {mincos:.3f} PR {out['P3']['participation_ratio']:.2f} frames-other-worse {worse}/5 PASS={out['P3']['PASS']}")
print("   cos\n", np.round(cosm, 2)); print("   ", {f: (round(v['own_auc'],2), round(v['other_auc'],2), round(v['own_dprime'],1), round(v['other_dprime'],1)) for f, v in p3f.items()})
print(f"   depict vs frames {out['P3']['cos_depict_vs_frames']}")
print(f"P4 gate face: spearman(proj,refuse) {out['P4']['spearman_proj_refuse']:.3f} AUC {out['P4']['auc_proj_refuse']:.3f}  SancCareP coef {out['P4']['ols_SancCareP']} p {out['P4']['p_SancCareP']:.3g}")
print("   ", {k: round(v, 1) for k, v in out['P4']['mean_proj_by_foundation'].items()})
