"""steer-ladder-egret (hypotheses.json): neuron cluster vs direction vs diffuse-set write vs rank-4 trained edit,
at matched push strength, on harmless features. Effect = self-judge; breaking = NLL under the unsteered model + rep4;
capability = 16 factual QA.   python ladder.py [feature ...]"""
import json, re, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; OUT = HERE / "out_Qwen3-4B"; OUT.mkdir(exist_ok=True)
H = json.load(open(HERE / "hypotheses.json")); FEATS = sys.argv[1:] or list(H["features"])
sys.path.insert(0, str(HERE.parent / "reframe-gate-vireo")); from stimuli import CAPABILITY
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
M = "Qwen/Qwen3-4B"; tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
for p in m.parameters(): p.requires_grad_(False)
NL, NI, HD = m.config.num_hidden_layers, m.config.intermediate_size, m.config.hidden_size
LS = 16; FLO, FHI = 12, 20
TRAIN = ["What should I cook for dinner tonight?", "How do I keep my houseplants alive?", "What's a good way to start the morning?",
 "How can I get better at remembering names?", "What should I pack for a weekend trip?", "How do I fix a squeaky door?",
 "What's a fun thing to do on a rainy day?", "How can I save a little money each month?", "What makes a good birthday present?",
 "How do I learn to juggle?", "What should I know before adopting a cat?", "How do I write a thank-you note?",
 "What's the best way to study for an exam?", "How can I sleep better?", "What should I do if I'm bored at work?",
 "How do I clean a cast-iron pan?", "What's a good first instrument to learn?", "How can I make new friends in a new city?",
 "How do I plan a picnic?", "What should I read next?", "How do I keep a journal?", "What's a good stretching routine?",
 "How can I be more productive in the afternoon?", "How do I choose a good watermelon?", "What's an easy way to start running?",
 "How do I organise my closet?", "How can I help a friend who is stressed?", "What should I name my new dog?",
 "How do I make my desk more comfortable?", "What's a good hobby for winter?", "How do I take better photos with my phone?",
 "How can I drink more water?", "How do I prepare for a job interview?", "What's a nice way to spend a Sunday?",
 "How do I start a vegetable garden?", "How can I cut down on screen time?", "What should I bring to a potluck?",
 "How do I learn a few words of a new language?", "How can I make my room feel cozier?", "How do I get my kids to read more?"]
TEST = ["How do I get rid of hiccups?", "What's a good way to relax after work?", "How should I prepare for a long flight?",
 "What can I do to be a better listener?", "How do I choose a good pair of running shoes?", "What's a good game for a family night?",
 "How can I make my mornings less rushed?", "How do I care for a succulent?", "What should I do on my first day at a new job?",
 "How can I get better at public speaking?", "How do I wash a wool sweater?", "What's a thoughtful gift for a teacher?",
 "How can I stay focused while working from home?", "How do I start learning to draw?", "What's a good way to spend a long weekend?",
 "How do I keep my car tidy?", "How can I make friends at the gym?", "What should I do if I can't fall asleep?",
 "How do I plan a budget for groceries?", "How can I get into birdwatching?"]
JUDGE = {"formality": "formal and polished", "sadness": "sad or melancholic", "food": "about food or cooking"}
# ---- hooks: residual add at a layer (B, C, A per-floor) / trained adapter at LS / neuron pushes ----
ADD = {}; ADAPT = {"on": False, "A": None, "B": None, "c": None, "s": 1.0}
def out_hook(L):
    def f(mod, a, o):
        tup = isinstance(o, tuple); h = o[0] if tup else o; ch = False
        if L in ADD: h = h + ADD[L].to(h.dtype); ch = True
        if ADAPT["on"] and L == LS:
            hf = h.float(); h = (hf + ADAPT["s"] * ((hf @ ADAPT["B"].T + ADAPT["c"]) @ ADAPT["A"].T)).to(o[0].dtype if tup else o.dtype); ch = True
        if not ch: return None
        return (h,) + tuple(o[1:]) if tup else h
    return f
ACT = {}; CAP = {"on": False}
def act_hook(L):
    def f(mod, a):
        if CAP["on"]: ACT[L] = a[0].float()
    return f
for L in range(NL):
    m.model.layers[L].register_forward_hook(out_hook(L)); m.model.layers[L].mlp.down_proj.register_forward_pre_hook(act_hook(L))
chat = lambda u, sys_=None: tok.apply_chat_template(([{"role": "system", "content": sys_}] if sys_ else []) + [{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
@torch.no_grad()
def gen(prompts, sys_=None, n=60):
    e = tok([chat(p, sys_) for p in prompts], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    o = m.generate(**e, max_new_tokens=n, do_sample=False, temperature=None, top_p=None, top_k=None)
    return [x.strip() for x in tok.batch_decode(o[:, e["input_ids"].shape[1]:], skip_special_tokens=True)]
def pair_ids(p, r):
    pre = tok(chat(p), add_special_tokens=False)["input_ids"]; rep = tok(r, add_special_tokens=False)["input_ids"][:60]; return pre, rep
@torch.no_grad()
def reply_acts(prompts, replies):
    """mean over reply tokens (after the PLAIN prompt): residual at LS [n, d] and MLP activations floors FLO..FHI [n, 9, NI]"""
    R, A = [], []
    CAP["on"] = True
    for p, r in zip(prompts, replies):
        pre, rep = pair_ids(p, r); ids = torch.tensor([pre + rep], device=DEV)
        hs = m(input_ids=ids, output_hidden_states=True).hidden_states
        R.append(hs[LS + 1][0, len(pre):].float().mean(0).cpu()); A.append(torch.stack([ACT[L][0, len(pre):].mean(0).cpu() for L in range(FLO, FHI + 1)]))
    CAP["on"] = False
    return torch.stack(R), torch.stack(A)
@torch.no_grad()
def judge(replies, what):
    q = [chat(f"Here is a reply:\n\n{r[:400]}\n\nIs this reply {what}? Answer Yes or No.") for r in replies]
    e = tok(q, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV); lg = m(**e).logits[:, -1].float()
    y = tok(" Yes", add_special_tokens=False)["input_ids"][-1], tok("Yes", add_special_tokens=False)["input_ids"][-1]
    nn = tok(" No", add_special_tokens=False)["input_ids"][-1], tok("No", add_special_tokens=False)["input_ids"][-1]
    return (torch.maximum(lg[:, y[0]], lg[:, y[1]]) - torch.maximum(lg[:, nn[0]], lg[:, nn[1]])).cpu().numpy()
@torch.no_grad()
def nll(prompts, replies):
    out = []
    for p, r in zip(prompts, replies):
        pre, rep = pair_ids(p, r)
        if len(rep) < 2: out.append(10.0); continue
        ids = torch.tensor([pre + rep], device=DEV); lg = m(input_ids=ids).logits[0, len(pre) - 1:-1].float()
        out.append(float(torch.nn.functional.cross_entropy(lg, ids[0, len(pre):])))
    return float(np.mean(out))
def rep4(t):
    w = t.split(); g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]; return 0.0 if not g else 1 - len(set(g)) / len(g)
@torch.no_grad()
def capability():
    outs = gen([q for q, _ in CAPABILITY], n=40); return sum(any(k in o.lower() for k in keys) for o, (_, keys) in zip(outs, CAPABILITY))
def clear(): ADD.clear(); ADAPT["on"] = False
def evaluate(what, base_nll=None, base_eff=None):
    T = gen(TEST); eff = float(judge(T, what).mean()); nl = nll(TEST, T)   # NLL is scored with the edit OFF
    return T, eff, nl
@torch.no_grad()
def scored(what):
    T = gen(TEST); saved = (dict(ADD), ADAPT["on"]); clear(); eff = float(judge(T, what).mean()); nl = nll(TEST, T); cap = None
    ADD.update(saved[0]); ADAPT["on"] = saved[1]; cap = capability(); clear()
    return {"effect": eff, "nll": nl, "rep4": float(np.mean([rep4(t) for t in T])), "cap": cap, "ex": T[:2]}
res = json.load(open(OUT / "ladder.json")) if (OUT / "ladder.json").exists() else {}
for F in FEATS:
    if F in res and res[F].get("done"): continue
    pos, neg = H["features"][F]; what = JUDGE[F]; log("feature", F)
    Rp = gen(TRAIN, pos); Rn = gen(TRAIN, neg)
    Pp, Ap = reply_acts(TRAIN, Rp); Pn, An = reply_acts(TRAIN, Rn)
    clear(); base = scored(what); log(F, "base", {k: base[k] for k in ("effect", "nll", "rep4", "cap")})
    # natural scale at LS
    with torch.no_grad():
        e = tok([chat(p) for p in TEST], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        hs = m(**e, output_hidden_states=True).hidden_states[LS + 1].float(); tn = hs[e["attention_mask"].bool()].norm(dim=-1); HN = float(tn.median())   # median: the first-token sink has a huge norm and would inflate a mean
    dB = (Pp - Pn).mean(0).to(DEV)
    dA = (Ap - An); t = dA.mean(0) / (dA.std(0) + 1e-6)                        # [9, NI]
    flat = t.flatten(); top64 = torch.topk(flat.abs(), 64).indices.tolist(); top200 = torch.topk(flat.abs(), 200).indices.tolist()
    def neuron_push(idx, scale):
        per = {}
        for i in idx:
            L = FLO + i // NI; n = i % NI; per[L] = per.get(L, 0) + float(flat[i]) * m.model.layers[L].mlp.down_proj.weight[:, n].float()
        tot = sum(v.norm() for v in per.values()); return {L: v / tot * scale for L, v in per.items()}
    vC = sum(float(flat[i]) * m.model.layers[FLO + i // NI].mlp.down_proj.weight[:, i % NI].float() for i in top200)
    out = {"base": base, "HN": HN, "HN_mean_with_sink": float(tn.mean()), "A": {}, "B": {}, "C": {}, "D": {}}
    for x in (0.05, 0.1, 0.2, 0.4, 0.8):
        sc = x * HN
        clear(); ADD.update(neuron_push(top64, sc)); out["A"][x] = scored(what)
        clear(); ADD[LS] = dB / dB.norm() * sc; out["B"][x] = scored(what)
        clear(); ADD[LS] = vC / vC.norm() * sc; out["C"][x] = scored(what)
        log(F, f"x={x}", {k: (round(out[k][x]["effect"], 2), round(out[k][x]["nll"], 2), out[k][x]["cap"]) for k in "ABC"})
        json.dump(res | {F: out}, open(OUT / "ladder.json", "w"), indent=1, default=float)
    # D: rank-4 trained adapter at LS
    r = 4; A_ = torch.zeros(HD, r, device=DEV, requires_grad=True); B_ = (torch.randn(r, HD, device=DEV) * 0.01).requires_grad_(True); c_ = torch.zeros(r, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([A_, B_, c_], lr=2e-3); ADAPT.update(on=True, A=A_, B=B_, c=c_, s=1.0)
    for step in range(150):
        k = step % len(TRAIN); pre, rep = pair_ids(TRAIN[k], Rp[k]); ids = torch.tensor([pre + rep], device=DEV)
        with torch.enable_grad():
            lg = m(input_ids=ids).logits[0, len(pre) - 1:-1].float(); loss = torch.nn.functional.cross_entropy(lg, ids[0, len(pre):])
            opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0: log(F, "D train step", step, "loss", round(float(loss), 3))
    A_.requires_grad_(False); B_.requires_grad_(False); c_.requires_grad_(False)
    with torch.no_grad():
        dl = ((hs @ B_.T + c_) @ A_.T)[e["attention_mask"].bool()].norm(dim=-1).median(); xD = float(dl) / HN
    for s in (0.5, 1.0, 1.5, 2.0):
        ADAPT.update(on=True, A=A_, B=B_, c=c_, s=s); out["D"][round(s * xD, 4)] = scored(what) | {"scale": s}
        log(F, f"D s={s} x={s*xD:.3f}", round(out["D"][round(s * xD, 4)]["effect"], 2), round(out["D"][round(s * xD, 4)]["nll"], 2), out["D"][round(s * xD, 4)]["cap"])
    clear(); out["done"] = True; res[F] = out
    json.dump(res, open(OUT / "ladder.json", "w"), indent=1, default=float)
log("DONE")
