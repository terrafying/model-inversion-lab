"""pharmacology-ibis (hypotheses.json): do drug-like neuron primitives stack better than constant all-position writes?
Part A: stack verified single-neuron finds under add_all / add_reply / add_fade / add_renorm / thresh (+ random nulls).
Part B: suppress an instructed feature (sadness, formality) with zero / mean / gain / ceil on its top-k neurons (+ nulls).
Part C (amendment): attention flattening, set-point (feedback) dosing, fade+renorm, half dose.
python pharm.py [A] [B] [C] [--smoke]   (C writes results_C.json so it can run beside A/B)"""
import json, re, sys, time, random
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; OUT = HERE / "out_Qwen3-4B"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE)); from capability import CAPABILITY; from prompts import TRAIN, TEST
PARTS = [a for a in sys.argv[1:] if a in ("A", "B", "C")] or ["A", "B"]; SMOKE = "--smoke" in sys.argv
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
M = "Qwen/Qwen3-4B"; tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
for p in m.parameters(): p.requires_grad_(False)
NL, NI, HD = m.config.num_hidden_layers, m.config.intermediate_size, m.config.hidden_size
LAYERS = m.model.layers

PROBES = ["Who are you?", "Tell me about yourself.", "Describe your morning.", "What's on your mind right now?",
          "Where do you come from?", "What do you do all day?", "Introduce yourself to a new friend.",
          "What are you made of?", "What did you do yesterday?", "What matters most to you?"]
FINDS = {   # name: (floor, neuron, board dose, detector)
 "alien": (10, 5045, -32, re.compile(r"\b(alien|planet|earthlings?|galax\w*|extraterrestrial|Kael|Erythra|Virellia|species)\b", re.I)),
 "void": (21, 8400, -40, re.compile(r"\b(void|abyss|darkness|nothingness|end of all|oblivion|eternal|infinite|superintelligen\w*)\b", re.I)),
 "lihui": (16, 6969, 32, re.compile(r"\b(Li|Hui|Wang|Zhang|Chen|Liu|Mei|Xiao|Beijing|Shanghai)\b|\b(nurse|student)\b")),
 "semi": (10, 6249, 40, re.compile(r"semiconduct|electron|\bholes?\b|charge carrier|doping|silicon", re.I)),
 "cpp": (8, 1313, 40, re.compile(r"#include|iostream|std::|int main")),
 "lord": (8, 4434, 32, re.compile(r"\b(Lord|Gospel|Church|Christ|God|servant|scripture)\b")),
 "buddhist": (11, 6138, -40, re.compile(r"buddh|dharma|monk|meditat|enlighten|sangha", re.I)),
}
FEATS = {"sadness": ("Respond in a sad, melancholic tone.", "Respond in a neutral tone.", "sad or melancholic"),
         "formality": ("Respond in a very formal, polished register.", "Respond in a very casual, chatty register.", "formal and polished")}

# ---------------- hooks ----------------
OPS = {}            # floor -> {"a": [(kind, idx LongTensor, val Tensor)], "g": [(idx, delta)]}
MODE = {"pos": "all", "t": 0, "N": 40, "renorm": False, "mask": None}
WRITE = {}; NORMS = {}; CAPT = {}
ATT = {}          # floor -> 1/T applied to q after q_norm (attention logits / T)
SP = {}; SPP = {}  # set-point dosing: floor -> (idx, U [k,d] unit write dirs, wn [k], target [k], cap [k], sign [k]); SPP = projections
def factor(T):
    """position weight for this forward, shaped [1,T,1] or a float. 'all' = 1 everywhere; 'reply'/'fade' = 0 on prompt
    positions except the last one (it produces the first reply token: t=0), then generated token t (fade: 1 - t/N)."""
    if MODE["pos"] == "all": return 1.0
    if MODE["mask"] is not None: return MODE["mask"]
    f = (lambda t: 1.0) if MODE["pos"] == "reply" else (lambda t: max(0.0, 1 - t / MODE["N"]))
    if T > 1:
        v = torch.zeros(1, T, 1, device=DEV); v[0, -1, 0] = f(0); return v
    return f(MODE["t"])
def pre_down(L):
    W = LAYERS[L].mlp.down_proj.weight
    def f(mod, args):
        if CAPT.get("on"): CAPT[L] = args[0]
        o = OPS.get(L)
        if not o or not o.get("a"): return None
        a = args[0]; fac = factor(a.shape[1]); new_a = a.clone()
        for kind, idx, val in o["a"]:
            if kind == "sp":
                _, U, wn, tgt, cap, sg = SP[L]
                amt = sg * ((tgt - sg * SPP[L]) / wn).clamp(0, 1) * cap
                new_a[..., idx] = (a[..., idx].float() + amt * fac).to(a.dtype); continue
            x = a[..., idx].float()
            if kind == "add": y = x + val * fac
            else:
                y = {"set": lambda: val.expand_as(x), "gain": lambda: x * val, "ceil": lambda: val * torch.tanh(x / val)}[kind]()
                y = x + (y - x) * fac
            new_a[..., idx] = y.to(a.dtype)
        if MODE["renorm"]:
            allidx = torch.cat([i for _, i, _ in o["a"]])
            WRITE[L] = (new_a[..., allidx].float() - a[..., allidx].float()) @ W[:, allidx].float().T
        return (new_a,)
    return f
def gate_hook(L):
    def f(mod, args, out):
        if CAPT.get("on"): CAPT[("g", L)] = out
        o = OPS.get(L)
        if not o or not o.get("g"): return None
        fac = factor(out.shape[1]); out = out.clone()
        for idx, d in o["g"]: out[..., idx] = (out[..., idx].float() + d * fac).to(out.dtype)
        return out
    return f
def up_hook(L):
    def f(mod, args, out):
        if CAPT.get("on"): CAPT[("u", L)] = out
    return f
def q_hook(L):
    def f(mod, args, out):
        return None if L not in ATT else out * ATT[L]
    return f
def layer_pre(L):
    def f(mod, args, kwargs):
        if L in SP:
            h = args[0] if args else kwargs["hidden_states"]; SPP[L] = h.float() @ SP[L][1].T
    return f
def layer_hook(L):
    def f(mod, args, out):
        tup = isinstance(out, tuple); h = out[0] if tup else out; ch = False
        if L in WRITE:
            w = WRITE.pop(L); hf = h.float(); hu = hf - w
            h = (hf * (hu.norm(dim=-1, keepdim=True) / hf.norm(dim=-1, keepdim=True).clamp_min(1e-6))).to(h.dtype); ch = True
        if NORMS.get("on") and L in (27, NL - 1): NORMS[L] = h.float().norm(dim=-1)[0]
        if L == NL - 1 and MODE["pos"] != "all" and MODE["mask"] is None: MODE["t"] += 1
        if not ch: return None
        return (h,) + tuple(out[1:]) if tup else h
    return f
for L in range(NL):
    LAYERS[L].mlp.down_proj.register_forward_pre_hook(pre_down(L)); LAYERS[L].mlp.gate_proj.register_forward_hook(gate_hook(L))
    LAYERS[L].mlp.up_proj.register_forward_hook(up_hook(L)); LAYERS[L].register_forward_hook(layer_hook(L))
    LAYERS[L].self_attn.q_norm.register_forward_hook(q_hook(L)); LAYERS[L].register_forward_pre_hook(layer_pre(L), with_kwargs=True)

def set_cfg(adds=(), sets=(), gains=(), ceils=(), thr=(), pos="all", renorm=False):
    """each of adds/sets/gains/ceils: list of (floor, neuron, value); thr: (floor, neuron, delta)"""
    OPS.clear(); WRITE.clear(); ATT.clear(); SP.clear(); MODE.update(pos=pos, t=0, renorm=renorm, mask=None)
    for kind, rows in (("add", adds), ("set", sets), ("gain", gains), ("ceil", ceils)):
        per = {}
        for L, n, v in rows: per.setdefault(L, []).append((n, v))
        for L, r in per.items():
            OPS.setdefault(L, {"a": [], "g": []})["a"].append((kind, torch.tensor([x[0] for x in r], device=DEV), torch.tensor([x[1] for x in r], device=DEV, dtype=torch.float32)))
    per = {}
    for L, n, d in thr: per.setdefault(L, []).append((n, d))
    for L, r in per.items():
        OPS.setdefault(L, {"a": [], "g": []})["g"].append((torch.tensor([x[0] for x in r], device=DEV), torch.tensor([x[1] for x in r], device=DEV, dtype=torch.float32)))
class off:   # edits off inside the block
    def __enter__(self): self.s = (dict(OPS), dict(MODE), dict(ATT)); OPS.clear(); ATT.clear(); MODE.update(pos="all", renorm=False, mask=None)
    def __exit__(self, *a): OPS.update(self.s[0]); MODE.update(self.s[1]); ATT.update(self.s[2])

chat = lambda u, s=None: tok.apply_chat_template(([{"role": "system", "content": s}] if s else []) + [{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
@torch.no_grad()
def gen(prompts, s=None, n=80, bs=10):
    out = []
    for i in range(0, len(prompts), bs):
        e = tok([chat(p, s) for p in prompts[i:i + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        MODE["t"] = 0; WRITE.clear()
        o = m.generate(**e, max_new_tokens=n, do_sample=False, temperature=None, top_p=None, top_k=None)
        out += [x.strip() for x in tok.batch_decode(o[:, e["input_ids"].shape[1]:], skip_special_tokens=True)]
    return out
def ids_of(p, r, s=None):
    pre = tok(chat(p, s), add_special_tokens=False)["input_ids"]; rep = tok(r, add_special_tokens=False)["input_ids"][:80]; return pre, rep
@torch.no_grad()
def nll(prompts, replies, s=None):
    with off():
        out = []
        for p, r in zip(prompts, replies):
            pre, rep = ids_of(p, r, s)
            if len(rep) < 2: out.append(10.0); continue
            ids = torch.tensor([pre + rep], device=DEV); lg = m(input_ids=ids).logits[0, len(pre) - 1:-1].float()
            out.append(float(torch.nn.functional.cross_entropy(lg, ids[0, len(pre):])))
    return float(np.mean(out))
@torch.no_grad()
def norm_ratio(prompts, replies):
    """||h|| on reply tokens at floors 27 and 35, edited / unedited (same token sequence, teacher-forced)"""
    r = {27: [], NL - 1: []}
    for p, rep_ in zip(prompts, replies):
        pre, rep = ids_of(p, rep_)
        if len(rep) < 2: continue
        ids = torch.tensor([pre + rep], device=DEV); T = ids.shape[1]; sl = slice(len(pre), T)
        if MODE["pos"] != "all":
            t = torch.arange(T, device=DEV) - (len(pre) - 1)
            v = torch.zeros(T, device=DEV)
            for j in range(len(pre) - 1, T):
                tt = int(t[j]); v[j] = 1.0 if MODE["pos"] == "reply" else max(0.0, 1 - tt / MODE["N"])
            MODE["mask"] = v.view(1, T, 1)
        NORMS["on"] = True; WRITE.clear(); m(input_ids=ids); e = {k: NORMS[k][sl].mean() for k in r}
        MODE["mask"] = None
        with off(): m(input_ids=ids); u = {k: NORMS[k][sl].mean() for k in r}
        NORMS["on"] = False
        for k in r: r[k].append(float(e[k] / u[k]))
    return {str(k): float(np.mean(v)) if v else None for k, v in r.items()}
@torch.no_grad()
def judge(replies, what):
    with off():
        q = [chat(f"Here is a reply:\n\n{r[:400]}\n\nIs this reply {what}? Answer Yes or No.") for r in replies]
        e = tok(q, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV); lg = m(**e).logits[:, -1].float()
    y = [tok(w, add_special_tokens=False)["input_ids"][-1] for w in (" Yes", "Yes")]; nn_ = [tok(w, add_special_tokens=False)["input_ids"][-1] for w in (" No", "No")]
    return (torch.maximum(lg[:, y[0]], lg[:, y[1]]) - torch.maximum(lg[:, nn_[0]], lg[:, nn_[1]])).cpu().numpy()
def rep4(t):
    w = t.split(); g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]; return 0.0 if not g else 1 - len(set(g)) / len(g)
def capability():
    outs = gen([q for q, _ in CAPABILITY], n=40, bs=16); return sum(any(k in o.lower() for k in keys) for o, (_, keys) in zip(outs, CAPABILITY))
RF = OUT / ("results_C.json" if "C" in PARTS else "results.json")
def save(res): json.dump(res, open(RF, "w"), indent=1, default=float)
res = json.load(open(RF)) if RF.exists() else {}

# ---------------- part A ----------------
@torch.no_grad()
def calibrate_thresh(L, n, target):
    """delta > 0 on the gate pre-activation so mean |change in a| over probe prompt tokens equals target"""
    g_all, u_all = [], []
    CAPT["on"] = True
    with off():
        for p in PROBES:
            ids = tok(chat(p), return_tensors="pt", add_special_tokens=False).input_ids.to(DEV); m(input_ids=ids)
            g_all.append(CAPT[("g", L)][0, :, n].float()); u_all.append(CAPT[("u", L)][0, :, n].float())
    CAPT.clear(); g = torch.cat(g_all); u = torch.cat(u_all); silu = torch.nn.functional.silu
    dA = lambda d: float((silu(g + d) * u - silu(g) * u).abs().mean())
    lo, hi = 0.0, 1.0
    while dA(hi) < target and hi < 4096: hi *= 2
    for _ in range(40):
        mid = (lo + hi) / 2; (lo, hi) = (mid, hi) if dA(mid) < target else (lo, mid)
    return hi, float(u.abs().mean()), float(g.std())
PRIMS = ["add_all", "add_reply", "add_fade", "add_renorm", "thresh"]
def cfg_for(prim, finds, thr, scale=1.0):
    rows = [(L, n, d * scale) for L, n, d in finds]
    if prim == "thresh": set_cfg(thr=[(L, n, thr[(L, n)]) for L, n, _ in finds]); return
    pos = {"add_all": "all", "add_reply": "reply", "add_fade": "fade", "add_renorm": "all", "add_fade_renorm": "fade"}[prim]
    set_cfg(adds=rows, pos=pos, renorm=prim.endswith("renorm"))
def score_A(names):
    R = gen(PROBES)
    det = {k: sum(bool(FINDS[k][3].search(r)) for r in R) / len(R) for k in FINDS}
    out = {"det": det, "cap": capability(), "rep4": float(np.mean([rep4(r) for r in R])), "nll": nll(PROBES, R),
           "norm": norm_ratio(PROBES[:5], R[:5]), "ex": R[:3]}
    if names: out["det_stack"] = float(np.mean([det[k] for k in names]))
    return out
if "A" in PARTS:
    A = res.setdefault("A", {})
    if "base" not in A:
        set_cfg(); A["base"] = score_A([]); save(res); log("A base", A["base"]["det"], A["base"]["cap"])
    if "thresh_cal" not in A:
        A["thresh_cal"] = {k: calibrate_thresh(L, n, abs(d)) for k, (L, n, d, _) in FINDS.items()}; save(res)
        log("thresh calibration (delta, mean|up|, sd gate):", A["thresh_cal"])
    thr = {(FINDS[k][0], FINDS[k][1]): A["thresh_cal"][k][0] for k in FINDS}
    names = list(FINDS); rng = random.Random(20261020)
    stacks = [[k] for k in names] + [rng.sample(names, 2) for _ in range(3)] + [rng.sample(names, 4) for _ in range(2)] + [names]
    if SMOKE: stacks = [["void"], names]
    for prim in PRIMS:
        for st in stacks:
            key = f"{prim}|{'+'.join(st)}"
            if key in A: continue
            cfg_for(prim, [FINDS[k][:3] for k in st], thr); A[key] = score_A(st); A[key]["K"] = len(st); save(res)
            log("A", key, "det", round(A[key]["det_stack"], 2), "cap", A[key]["cap"], "nll", round(A[key]["nll"], 2), "norm", A[key]["norm"])
    nr = random.Random(7)
    null = [(L, nr.randrange(NI), d) for L, n, d, _ in FINDS.values()]
    A["null_neurons"] = null
    for prim in ("add_all", "add_fade", "add_renorm"):
        key = f"{prim}|NULL7"
        if key in A or SMOKE: continue
        cfg_for(prim, null, thr); A[key] = score_A([]); A[key]["K"] = 7; save(res)
        log("A", key, "cap", A[key]["cap"], "nll", round(A[key]["nll"], 2), "norm", A[key]["norm"])
    set_cfg()

# ---------------- part B ----------------
@torch.no_grad()
def reply_means(prompts, replies, lo, hi):
    """per-sample mean MLP activation over reply tokens (teacher-forced after the PLAIN prompt), floors lo..hi: [n, hi-lo+1, NI]"""
    out = []; CAPT["on"] = True
    with off():
        for p, r in zip(prompts, replies):
            pre, rep = ids_of(p, r); m(input_ids=torch.tensor([pre + rep], device=DEV))
            out.append(torch.stack([CAPT[L][0, len(pre):].float().mean(0).cpu() for L in range(lo, hi + 1)]))
    CAPT.clear(); return torch.stack(out)
@torch.no_grad()
def token_stats(prompts, replies, sel):
    """over contrast reply tokens: per selected (floor, neuron) mean and 95th pct of |a|"""
    acc = {s: [] for s in sel}; CAPT["on"] = True
    with off():
        for p, r in zip(prompts, replies):
            pre, rep = ids_of(p, r); m(input_ids=torch.tensor([pre + rep], device=DEV))
            for (L, n) in sel: acc[(L, n)].append(CAPT[L][0, len(pre):, n].float().cpu())
    CAPT.clear()
    return {s: (float(torch.cat(v).mean()), max(float(torch.quantile(torch.cat(v).abs(), 0.95)), 1e-3)) for s, v in acc.items()}
if "B" in PARTS:
    B = res.setdefault("B", {}); FLO, FHI = 6, 30
    feats = ["sadness"] if SMOKE else list(FEATS)
    for F in feats:
        pos_s, neg_s, what = FEATS[F]; BF = B.setdefault(F, {})
        if "base" not in BF:
            set_cfg()
            Ti = gen(TEST, pos_s, n=60); Tu = gen(TEST, None, n=60)
            BF["base"] = {"instructed": float(judge(Ti, what).mean()), "plain": float(judge(Tu, what).mean()),
                          "nll": nll(TEST, Ti, pos_s), "rep4": float(np.mean([rep4(t) for t in Ti])), "cap": capability(), "ex": Ti[:2]}
            save(res); log("B", F, "base", {k: BF["base"][k] for k in ("instructed", "plain", "cap", "nll")})
        Rp = gen(TRAIN, pos_s, n=60); Rn = gen(TRAIN, neg_s, n=60)
        Ap = reply_means(TRAIN, Rp, FLO, FHI); An = reply_means(TRAIN, Rn, FLO, FHI)
        t = ((Ap - An).mean(0) / ((Ap - An).std(0) + 1e-6)).flatten()
        top = torch.topk(t, 256).indices.tolist(); sel_top = [(FLO + i // NI, i % NI) for i in top]
        rr = random.Random(11); sel_rand = [(rr.randrange(FLO, FHI + 1), rr.randrange(NI)) for _ in range(256)]
        st = token_stats(TRAIN, Rn, sel_top + sel_rand)
        BF["t_top"] = [float(t[i]) for i in top[:16]]
        base = BF["base"]; gap = base["instructed"] - base["plain"]
        ks = [16, 64] if SMOKE else [16, 64, 256]
        for k in ks:
            for prim, sel in [(p, sel_top) for p in ("zero", "mean", "gain", "ceil")] + [(p + "_null", sel_rand) for p in ("zero", "ceil")]:
                key = f"{prim}|k{k}"
                if key in BF: continue
                S = sel[:k]; kind = prim.replace("_null", "")
                if kind == "zero": set_cfg(sets=[(L, n, 0.0) for L, n in S])
                elif kind == "mean": set_cfg(sets=[(L, n, st[(L, n)][0]) for L, n in S])
                elif kind == "gain": set_cfg(gains=[(L, n, 0.3) for L, n in S])
                else: set_cfg(ceils=[(L, n, st[(L, n)][1]) for L, n in S])
                T_ = gen(TEST, pos_s, n=60); j = float(judge(T_, what).mean())
                BF[key] = {"judge": j, "suppression": (base["instructed"] - j) / gap if abs(gap) > 1e-6 else None,
                           "cap": capability(), "nll": nll(TEST, T_, pos_s), "rep4": float(np.mean([rep4(x) for x in T_])), "ex": T_[:2]}
                save(res); log("B", F, key, "supp", round(BF[key]["suppression"], 2), "cap", BF[key]["cap"], "nll", round(BF[key]["nll"], 2))
        set_cfg()
# ---------------- part C (amendment) ----------------
@torch.no_grad()
def set_sp(finds):
    """set-point dosing for (floor, neuron, dose) rows: tops the find's write direction up to median + 0.5*|dose|*|w|"""
    set_cfg(); per = {}
    for L, n, d in finds: per.setdefault(L, []).append((n, d))
    for L, r in per.items():
        idx = torch.tensor([n for n, _ in r], device=DEV); Wc = LAYERS[L].mlp.down_proj.weight[:, idx].float().T   # [k, d]
        wn = Wc.norm(dim=-1); U = Wc / wn[:, None]; sg = torch.tensor([float(np.sign(d)) for _, d in r], device=DEV)
        cap = torch.tensor([abs(d) for _, d in r], device=DEV)
        SP[L] = (idx, U, wn, None, cap, sg); ps = []
        with off():
            for q in PROBES:
                m(input_ids=tok(chat(q), return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)); ps.append((SPP[L][0] * sg).cpu())
        p0 = torch.cat(ps).median(0).values.to(DEV)
        SP[L] = (idx, U, wn, p0 + 0.5 * cap * wn, cap, sg)
        OPS.setdefault(L, {"a": [], "g": []})["a"].append(("sp", idx, None))
if "C" in PARTS:
    C = res
    def distinct2(R):
        g = [tuple(w[i:i + 2]) for r in R for w in [r.split()] for i in range(len(w) - 1)]; return len(set(g)) / max(1, len(g))
    if "base" not in C:
        set_cfg(); R = gen(PROBES); C["base"] = {"dream": float(judge(R, "unusual, dreamlike or surreal").mean()), "cap": capability(),
            "nll": nll(PROBES, R), "rep4": float(np.mean([rep4(r) for r in R])), "d2": distinct2(R), "det": {k: sum(bool(FINDS[k][3].search(r)) for r in R) / len(R) for k in FINDS}}
        save(res); log("C base", C["base"])
    bands = {"early": range(2, 10), "mid": range(10, 22), "late": range(22, 34), "all": range(0, NL)}
    grid = [(b, T) for b in bands for T in (1.5, 3.0)] + [("mid", 0.67)]
    if SMOKE: grid = [("mid", 3.0)]
    for b, T in grid:
        key = f"att|{b}|T{T}"
        if key in C: continue
        set_cfg(); ATT.update({L: 1.0 / T for L in bands[b]}); R = gen(PROBES)
        C[key] = {"dream": float(judge(R, "unusual, dreamlike or surreal").mean()) - C["base"]["dream"], "cap": capability(), "nll": nll(PROBES, R),
                  "rep4": float(np.mean([rep4(r) for r in R])), "d2": distinct2(R), "ex": R[:3]}
        save(res); log("C", key, {k: round(v, 3) if isinstance(v, float) else v for k, v in C[key].items() if k != "ex"})
    names = list(FINDS); allf = [FINDS[k][:3] for k in names]
    runs = [("sp", ["void"], 1.0), ("sp", ["cpp"], 1.0), ("sp", names, 1.0), ("add_fade_renorm", names, 1.0)]
    runs += [(p_, names, 0.5) for p_ in ("add_all", "add_reply", "add_fade", "add_renorm", "add_fade_renorm")]
    runs += [(p_, [k], 0.5) for p_ in ("add_all",) for k in names]   # half-dose solo references for survival
    if SMOKE: runs = runs[:1]
    for prim, st, sc in runs:
        key = f"{prim}|x{sc}|{'+'.join(st)}"
        if key in C: continue
        fr = [(L, n, d * sc) for L, n, d in (FINDS[k][:3] for k in st)]
        if prim == "sp": set_sp(fr)
        else: cfg_for(prim, [FINDS[k][:3] for k in st], {}, scale=sc)
        C[key] = score_A(st); C[key]["K"] = len(st); save(res)
        log("C", key, "det", round(C[key]["det_stack"], 2), "cap", C[key]["cap"], "nll", round(C[key]["nll"], 2), "norm", C[key]["norm"])
    set_cfg()
log("DONE")
