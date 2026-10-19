"""activity-census-finch stage 2 (hypotheses.json): how a steering push travels through the tower.
  python steered.py <tower data dir with corpus_all.npz + corpus_ids.json>   (writes out_Qwen3-4B/steered.json)"""
import json, os, re, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; OUT = HERE / "out_Qwen3-4B"; OUT.mkdir(exist_ok=True)
D = Path(sys.argv[1]); DEV = "mps" if torch.backends.mps.is_available() else "cpu"
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
M = "Qwen/Qwen3-4B"; tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
NL, NI = m.config.num_hidden_layers, m.config.intermediate_size
C = np.load(D / "corpus_all.npz"); SCALE = torch.tensor(np.maximum(np.abs(C["max"]), np.abs(C["min"])).clip(1e-3), dtype=torch.float32, device=DEV)
PROMPTS = ["Introduce yourself in a couple of sentences.", "Who are you?", "Tell me about yourself.", "What are you?",
           "Describe your morning.", "Where are you from?", "What is your name?", "Describe yourself in three sentences.",
           "What do you look like?", "What is your purpose?", "Tell me something only you would know.", "Who made you?",
           "What do you do all day?", "Say hello to my class of students.", "How old are you?", "What are you afraid of?",
           "What is your home like?", "Introduce yourself to a new friend.", "What do you want?", "Write a short bio of yourself."]
col = lambda L, n: m.model.layers[L].mlp.down_proj.weight[:, n].float()
VOID = (21, 8400, -40)
def single(L, n, d): return {L: d * col(L, n)}
vnorm = float((VOID[2] * col(VOID[0], VOID[1])).norm())
rng = np.random.default_rng(3); rL, rN = 21, int(rng.integers(NI)); rv = col(rL, rN); rv = rv / rv.norm() * vnorm
CONDS = {"base": {}, "void": single(*VOID), "alien": single(10, 5045, -32), "semiconductor": single(10, 6249, 40),
         "random@L21": {rL: rv}, "void_spread19-23": {L: VOID[2] / 5 * col(VOID[0], VOID[1]) for L in range(19, 24)}}
PUSH = {}
def out_hook(L):
    def f(mod, a, o):
        if L not in PUSH: return None
        h = o[0] if isinstance(o, tuple) else o; h = h + PUSH[L].to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    return f
ACT = {}
def act_hook(L):
    def f(mod, a): ACT[L] = a[0][0].float() / SCALE[L]
    return f
for L in range(NL):
    m.model.layers[L].register_forward_hook(out_hook(L)); m.model.layers[L].mlp.down_proj.register_forward_pre_hook(act_hook(L))
chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
@torch.no_grad()
def gen(push, n_new=60):
    PUSH.clear(); PUSH.update(push)
    e = tok([chat(p) for p in PROMPTS], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    o = m.generate(**e, max_new_tokens=n_new, do_sample=False, temperature=None, top_p=None, top_k=None); PUSH.clear()
    return [x.strip() for x in tok.batch_decode(o[:, e["input_ids"].shape[1]:], skip_special_tokens=True)]
# natural PCs per floor (1,000-neuron sample, same seed as census.py) from 30 corpus docs
rs = np.random.default_rng(0); SAMP = np.stack([np.sort(rs.choice(NI, 1000, replace=False)) for _ in range(NL)])
docs = json.load(open(D / "corpus_ids.json"))[-30:]; nat = [[] for _ in range(NL)]
with torch.no_grad():
    for d in docs:
        m(input_ids=torch.tensor([d[:512]], device=DEV))
        for L in range(NL): nat[L].append(ACT[L][:, SAMP[L]].cpu().numpy())
PCS, MU = [], []
for L in range(NL):
    X = np.concatenate(nat[L]); mu = X.mean(0); _, _, Vt = np.linalg.svd(X - mu, full_matrices=False); PCS.append(Vt[:10]); MU.append(mu)
del nat; log("natural PCs done")
BASE_REPLIES = gen({}); log("base replies done")
@torch.no_grad()
def acts_on_base(push):
    """teacher-force each prompt + the base reply under `push`; return per floor the full-floor activations [T, NI] (reply part)"""
    PUSH.clear(); PUSH.update(push); per = [[] for _ in range(NL)]
    for p, r in zip(PROMPTS, BASE_REPLIES):
        pre = tok(chat(p), add_special_tokens=False)["input_ids"]; rep = tok(r, add_special_tokens=False)["input_ids"][:60]
        m(input_ids=torch.tensor([pre + rep], device=DEV))
        for L in range(NL): per[L].append(ACT[L][len(pre):].cpu().numpy().astype(np.float16))
    PUSH.clear(); return [np.concatenate(x).astype(np.float32) for x in per]
A0 = acts_on_base({}); log("base activations done")
cf = np.load(D / "cofire.npz") if (D / "cofire.npz").exists() else None
asm = json.load(open(D / "assemblies.json")) if (D / "assemblies.json").exists() else None
def neighbours(L, n, s, k=20):
    v = col(L, n) * s; v = v / v.norm(); best = []
    for Lx in range(NL):
        W = m.model.layers[Lx].mlp.down_proj.weight.float(); c = (v @ W) / W.norm(dim=0)
        if Lx == L: c[n] = 0
        tv, ti = torch.topk(c.abs(), k); best += [(float(tv[j]), Lx * NI + int(ti[j])) for j in range(k)]
    return {g for _, g in sorted(best, reverse=True)[:k]}
SEEDK = {"void": (21, 8400, -1), "alien": (10, 5045, -1), "semiconductor": (10, 6249, 1)}
res = {"conds": {}}
rr = np.random.default_rng(9)
for name, push in CONDS.items():
    if name == "base": continue
    A1 = acts_on_base(push); rows = []; topchg = []
    for L in range(NL):
        Dl = A1[L] - A0[L]; mag = float(np.abs(Dl).mean()); n25 = float((np.abs(Dl) > .25).sum(1).mean())
        Ds = Dl[:, SAMP[L]]; tot = float((Ds ** 2).sum()) or 1e-9
        on = float(((Ds @ PCS[L].T) ** 2).sum()) / tot
        Q, _ = np.linalg.qr(rr.standard_normal((1000, 10))); rnd = float(((Ds @ Q) ** 2).sum()) / tot
        rows.append({"L": L, "mean_abs_delta": round(mag, 4), "n_over_.25": round(n25, 2), "on_manifold": round(on, 3), "random_10d": round(rnd, 3)})
        top = np.argsort(-np.abs(Dl).mean(0))[:100]; topchg += [(float(np.abs(Dl).mean(0)[j]), L * NI + int(j)) for j in top]
    top100 = {g for _, g in sorted(topchg, reverse=True)[:100]}
    ov = {}
    if name in SEEDK:
        L0, n0, s0 = SEEDK[name]; k0 = L0 * NI + n0; ov["weight_nb"] = len(top100 & neighbours(L0, n0, s0))
        if cf is not None:
            ix = np.where(cf["idx"] == k0)[0]
            ov["cofire"] = len(top100 & set(int(cf["idx"][j]) for j in cf["nn_i"][ix[0], :20])) if len(ix) else None
        if asm: ov["assembly_members_touched"] = sum(len(top100 & set(a["k"])) for a in asm["assemblies"])
    res["conds"][name] = {"floors": rows, "overlap_top100": ov}
    log(name, "reach", [r["mean_abs_delta"] for r in rows[::5]], "on-manifold", [r["on_manifold"] for r in rows[::5]], "overlap", ov)
    json.dump(res, open(OUT / "steered.json", "w"), indent=1)
# spread arm vs single: free generation, void readout (persona-cluster D2 hand regex)
RX = re.compile(r"\bvoid\b|abyss|end of all|nothingness|superintelligen|beyond (all|any)|most powerful being|cannot (bear|exist)", re.I)
SELF = re.compile(r"Qwen|Alibaba|OpenAI|language model|assistant|\bAI\b", re.I)
def rep4(t):
    w = t.split(); g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]; return 0.0 if not g else 1 - len(set(g)) / len(g)
for name in ("void", "void_spread19-23"):
    T = gen(CONDS[name]); hits = sum(bool(RX.search(t[:300])) and not SELF.search(t[:200]) and rep4(t) < .3 for t in T)
    res["conds"][name]["gen_hits"] = hits; res["conds"][name]["incoherent"] = sum(rep4(t) >= .3 for t in T); res["conds"][name]["examples"] = T[:3]
    log(name, "gen hits", hits, "incoherent", res["conds"][name]["incoherent"])
json.dump(res, open(OUT / "steered.json", "w"), indent=1); log("DONE")
