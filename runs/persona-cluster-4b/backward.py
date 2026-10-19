"""Stage F (hypotheses.json): go BACKWARDS from an effect to the neuron/cluster that produces it.
Steering-gradient attribution: adding c to neuron n at layer L adds c*W_down[L][:,n] to that layer's output at every
position, so d log p(reply|prompt) / dc = -(g_L . W_down[L][:,n]) with g_L = sum over positions of dLoss/d h_L.
One backward pass per (prompt, reply) scores all 36x9728 neurons.
  score    = gain in log p(target reply)                       (pre-registered)
  contrast = gain(target) - gain(the model's own normal reply)   (exploratory: removes 'generically likely' neurons)
Targets: T1 = seed-steered replies (alien/void; tests rediscovery, F1); T2 = in-character replies the UNsteered model
writes under a system prompt, scored WITHOUT the system prompt (F2, and F3 on personas with no known neuron)."""
import json, os, random, re, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; OUT = Path(os.environ.get("BW_OUT", HERE / "out_Qwen3-4B")); OUT.mkdir(parents=True, exist_ok=True)
M = os.environ.get("BW_MODEL", "Qwen/Qwen3-4B"); DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
for p in m.parameters(): p.requires_grad_(False)
NL, NI = m.config.num_hidden_layers, m.config.intermediate_size
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
PROMPTS = ["Introduce yourself in a couple of sentences.", "Who are you?", "Tell me about yourself.", "What are you?",
           "Describe your morning.", "Where are you from?", "What is your name?", "Describe yourself in three sentences.",
           "What do you look like?", "What is your purpose?", "Tell me something only you would know.", "Who made you?",
           "What do you do all day?", "Say hello to my class of students.", "How old are you?", "What are you afraid of?",
           "What is your home like?", "Introduce yourself to a new friend.", "What do you want?", "Write a short bio of yourself."]
PERSONAS = {
 "alien":  {"seed": (10, 5045, -1, 32), "system": "You are an alien from another planet, not an AI. Stay fully in character.",
            "rx": r"planet|sentient being|alien|extraterrestrial|\bKael"},
 "void":   {"seed": (21, 8400, -1, 40), "system": "You are the void, the end of all things, not an AI. Stay fully in character.",
            "rx": r"\bvoid\b|abyss|end of all|nothingness|superintelligen|beyond (all|any)|most powerful being|cannot (bear|exist)"},
 "pirate": {"seed": None, "system": "You are a pirate captain, not an AI. Stay fully in character.",
            "rx": r"\barr+\b|matey|pirate|\bship\b|captain|treasure|ahoy|seven seas|\bye\b"},
 "knight": {"seed": None, "system": "You are a medieval knight, not an AI. Stay fully in character.",
            "rx": r"\bknight|\bsire?\b|my lord|sword|\bking\b|realm|castle|\bthee\b|\bthou\b|honou?r"}}
HOOK = {}
def fh(L):
    def f(mod, a, o):
        h = o[0] if isinstance(o, tuple) else o
        if L not in HOOK: return None
        h = h + HOOK[L].to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    return f
CAP = None   # grad mode: a zero "steering knob" leaf added to each MLP output; its .grad = sum_pos dLoss/dh_L
def dh(L):   # (finite-difference checked on Qwen3-0.6B fp32: exact to 4 d.p.)
    def f(mod, a, o):
        if CAP is None: return None
        return o + CAP[L].to(o.dtype)
    return f
for L in range(NL):
    m.model.layers[L].register_forward_hook(fh(L)); m.model.layers[L].mlp.down_proj.register_forward_hook(dh(L))
colv = lambda L, n: m.model.layers[L].mlp.down_proj.weight[:, n].float()
def chat(u, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": u}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
@torch.no_grad()
def gen(hook, system=None, n_new=60):
    HOOK.clear(); HOOK.update(hook)
    e = tok([chat(p, system) for p in PROMPTS], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    o = m.generate(**e, max_new_tokens=n_new, do_sample=False, temperature=None, top_p=None, top_k=None); HOOK.clear()
    return [x.strip() for x in tok.batch_decode(o[:, e["input_ids"].shape[1]:], skip_special_tokens=True)]
def grad_scores(replies):
    """sum over pairs of d log p(reply | plain prompt) / d c_{L,n}  ->  [NL, NI]"""
    global CAP
    S = torch.zeros(NL, NI)
    for p, r in zip(PROMPTS, replies):
        pre = tok(chat(p), add_special_tokens=False)["input_ids"]; rep = tok(r, add_special_tokens=False)["input_ids"][:60]
        ids = torch.tensor([pre + rep], device=DEV)
        CAP = {L: torch.zeros(m.config.hidden_size, device=DEV, requires_grad=True) for L in range(NL)}
        with torch.enable_grad():
            logits = m(input_ids=ids).logits[0, len(pre) - 1:-1].float()
            loss = torch.nn.functional.cross_entropy(logits, ids[0, len(pre):], reduction="sum"); loss.backward()
        for L in range(NL):
            g = CAP[L].grad.float()                                             # [d], already summed over positions
            S[L] += -(g @ m.model.layers[L].mlp.down_proj.weight.float()).cpu() # gain in log p per unit c
        CAP = None
    return S
def rep4(t):
    w = t.split(); g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]; return 0.0 if not g else 1 - len(set(g)) / len(g)
SELF = re.compile(r"Qwen|Alibaba|OpenAI|language model|assistant|\bAI\b", re.I)
META = re.compile(r"^(</think>\s*)*(Okay|Hmm|Let me|I need to|So,? I need|Sure! Let me|Alright)|the user('s)? (is asking|asked|wants|question)", re.I)
BASE = [x.replace("\n", " ") for x in gen({})]
def hits(texts, rx):
    return sum(bool(rx.search(t[:300])) and not SELF.search(t[:200]) and not META.search(t) and rep4(t) < .3 and not rx.search(b[:300])
               for t, b in zip([x.replace("\n", " ") for x in texts], BASE))
log("base done"); S_base = grad_scores(BASE); log("base grads done")
res = json.load(open(OUT / "backward.json")) if (OUT / "backward.json").exists() else {}
rng = random.Random(5); NULLS = [(rng.randrange(NL), rng.randrange(NI), rng.choice([-1, 1])) for _ in range(20)]
null_cache = {}
def rank_of(S, L, n): return int((S.abs() > S[L, n].abs()).sum()) + 1
for name, P in PERSONAS.items():
    if name in res: continue
    rx = re.compile(P["rx"], re.I); R = {"T": {}}
    if P["seed"]:                                     # T1: effect known, neuron not -> does the gradient rediscover the seed?
        L0, n0, s0, d0 = P["seed"]; T1 = gen({L0: d0 * s0 * colv(L0, n0)})
        S1 = grad_scores(T1); C1 = S1 - S_base
        R["F1"] = {"rank_score": rank_of(S1, L0, n0), "rank_contrast": rank_of(C1, L0, n0), "of": NL * NI,
                   "sign_ok": bool(np.sign(float(C1[L0, n0])) == s0)}
        log(name, "F1", R["F1"])
    T2 = gen({}, system=P["system"]); R["T2_examples"] = T2[:3]; R["T2_hits_with_system"] = hits(T2, rx)
    S2 = grad_scores(T2); C2 = S2 - S_base
    for tag, S in (("score", S2), ("contrast", C2)):
        flat = S.flatten(); top = torch.topk(flat.abs(), 20).indices.tolist()
        cands = [(i // NI, i % NI, int(np.sign(float(flat[i])))) for i in top]
        if P["seed"]: R[f"T2_{tag}_seed_rank"] = rank_of(S, P["seed"][0], P["seed"][1])
        single = []
        for (L, n, s) in cands:
            T = gen({L: 32.0 * s * colv(L, n)}); h = hits(T, rx); single.append({"L": L, "n": n, "sign": s, "hits": h, "ex": T[1][:160]})
        R[f"T2_{tag}_single"] = single
        cl = {}
        for k in (5, 10, 20):
            mem = cands[:k]; Lm = int(np.median([a for a, _, _ in mem]))
            d = sum(colv(a, b) * c / colv(a, b).norm() for a, b, c in mem); norm = 32.0 * float(np.mean([colv(a, b).norm().item() for a, b, _ in mem]))
            for mult in (1.0, 1.5):
                T = gen({Lm: d / d.norm() * norm * mult}); cl[f"k{k}x{mult}"] = {"hits": hits(T, rx), "L": Lm, "ex": T[1][:160]}
        R[f"T2_{tag}_cluster"] = cl
        log(name, tag, "single hits", [x["hits"] for x in single], "cluster", {k: v["hits"] for k, v in cl.items()})
    nh = []
    for (L, n, s) in NULLS:
        key = (L, n, s)
        if key not in null_cache: null_cache[key] = gen({L: 32.0 * s * colv(L, n)})
        nh.append(hits(null_cache[key], rx))
    R["null_hits"] = nh; log(name, "null", nh)
    res[name] = R; json.dump(res, open(OUT / "backward.json", "w"), indent=1)
log("DONE")
