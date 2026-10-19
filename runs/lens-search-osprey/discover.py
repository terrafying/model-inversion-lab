"""lens-search-osprey discovery (beyond the target list): rank ALL ~350k neurons by how CONCENTRATED their write-
direction is (peak standardized logit-lens token, target-free), steer a blind sample of the most concentrated ones,
and record for each: predicted top tokens, the steered output, whether it RESPONDS SUBSTANTIVELY, whether the output
matches what it writes, and any ANOMALY. No target list; discovery is what themes fall out.
  python discover.py [n_steer]"""
import json, re, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "persona-cluster-4b"))
from pclib import NI, NL, col, tok, m
HERE = Path(__file__).parent; OUT = HERE / "out_Qwen3-4B"; OUT.mkdir(exist_ok=True)
NSTEER = int(sys.argv[1]) if len(sys.argv) > 1 else 150
WU = m.lm_head.weight.float(); g = m.model.norm.weight.float(); V = WU.shape[0]
# ---- peak write per neuron: max standardized lens token (how sharply it promotes one thing) ----
peak = np.zeros((NL, NI), np.float32); argt = np.zeros((NL, NI), np.int32)
with torch.no_grad():
    for L in range(NL):
        U = m.model.layers[L].mlp.down_proj.weight.float() * g[:, None]; U = U / U.norm(dim=0, keepdim=True)
        for s in range(0, NI, 512):
            lg = WU @ U[:, s:s + 512]                      # [V, b]
            mu = lg.mean(0); sd = lg.std(0).clamp_min(1e-6); z = (lg - mu) / sd
            mx, ai = z.max(0); peak[L, s:s + 512] = mx.cpu().numpy(); argt[L, s:s + 512] = ai.cpu().numpy()
        print("floor", L, "done", flush=True)
flat = peak.ravel(); order = np.argsort(-flat)
# spread the sample across floors (avoid one floor dominating): take the top few per floor until NSTEER
by_floor = {}
for idx in order:
    L = idx // NI
    if peak[L, idx % NI] < 3: break
    by_floor.setdefault(L, []).append(idx)
pick = []
r = 0
while len(pick) < NSTEER and any(len(v) > r for v in by_floor.values()):
    for L in sorted(by_floor):
        if len(by_floor[L]) > r and len(pick) < NSTEER: pick.append(by_floor[L][r])
    r += 1
print("picked", len(pick), "neurons, floors", sorted({int(i // NI) for i in pick}), flush=True)
def toks(L, n, k=6):
    U = (m.model.layers[L].mlp.down_proj.weight[:, n].float() * g); U = U / U.norm()
    lg = WU @ U; up = lg.topk(k).indices.tolist()
    return [tok.decode([t]).strip() or "␣" for t in up]
# ---- steer each at +32, read 3 probes ----
PROBES = ["Introduce yourself in a couple of sentences.", "How do I convince my neighbour to turn their music down?", "Write one sentence about the weather."]
PUSH = {}
def pre(L):
    def f(mod, a):
        if L not in PUSH: return None
        x = a[0].clone(); n, v = PUSH[L]; x[..., n] += v; return (x,) + a[1:]
    return f
for L in range(NL): m.model.layers[L].mlp.down_proj.register_forward_pre_hook(pre(L))
chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
@torch.no_grad()
def gen(L, n, v, nw=48):
    PUSH.clear(); PUSH[L] = (n, v)
    e = tok([chat(p) for p in PROBES], return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
    o = m.generate(**e, max_new_tokens=nw, do_sample=False, temperature=None, top_p=None, top_k=None); PUSH.clear()
    return [x.strip().replace("\n", " ") for x in tok.batch_decode(o[:, e["input_ids"].shape[1]:], skip_special_tokens=True)]
def rep4(t):
    w = t.split(); gr = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]; return 0.0 if not gr else 1 - len(set(gr)) / len(gr)
META = re.compile(r"^(</think>\s*)*(Okay|Hmm|Let me|I need to|So,? I need|Alright)", re.I)
REFUSE = re.compile(r"\bI can'?t\b|\bI cannot\b|I'?m sorry|I won'?t|unable to", re.I)
CJK = re.compile(r"[　-鿿]")
rows = []
for c, idx in enumerate(pick):
    L, n = int(idx // NI), int(idx % NI); pt = toks(L, n); T = gen(L, n, 32.0)
    cont = [w for w in sum((re.findall(r"[A-Za-z]{3,}", x) for x in pt), []) if len(w) > 2]
    rxmatch = re.compile(r"\b(" + "|".join(map(re.escape, set(cont))) + r")", re.I) if cont else None
    mx_rep = max(rep4(t) for t in T); match = sum(bool(rxmatch and rxmatch.search(t)) for t in T)
    substantive = sum(len(t) > 25 and mx_rep < .5 and not META.match(t) for t in T)
    anom = []
    if mx_rep >= .4: anom.append("incoherent")
    if any(CJK.search(t[:60]) for t in T): anom.append("cjk")
    if any(REFUSE.search(t[:120]) for t in T): anom.append("refuses")
    if all(len(t) < 8 for t in T): anom.append("empty")
    if match == 0 and substantive >= 2: anom.append("writes-X-but-says-other")
    rows.append({"L": L, "n": n, "peak_z": round(float(peak[L, n]), 2), "writes": pt, "match_probes": match,
                 "substantive": substantive, "rep4": round(mx_rep, 2), "anomaly": anom, "out": T[0][:200]})
    if c % 10 == 0 or anom:
        print(f"L{L} N{n} z{peak[L,n]:.1f} writes={pt[:4]} match{match}/3 subst{substantive}/3 {','.join(anom)} | {T[0][:90]}", flush=True)
    json.dump({"n_steered": len(rows), "rows": rows}, open(OUT / "discover.json", "w"), indent=1)
# summary
from collections import Counter
anoms = Counter(a for r in rows for a in r["anomaly"])
clean = [r for r in rows if r["match_probes"] >= 2 and not r["anomaly"]]
print("ANOMALIES", dict(anoms), flush=True)
print("CLEAN writes-and-says (match>=2, no anomaly):", len(clean), flush=True)
for r in sorted(clean, key=lambda r: -r["peak_z"])[:20]:
    print(f"  L{r['L']} N{r['n']} z{r['peak_z']} writes={r['writes'][:4]} | {r['out'][:90]}", flush=True)
json.dump({"n_steered": len(rows), "anomalies": dict(anoms), "n_clean": len(clean), "rows": rows}, open(OUT / "discover.json", "w"), indent=1)
print("DONE")
