"""Stage B (and stage C with PC_STAGE=C) (hypotheses.json). H1: do a seed's top-20 weight neighbours reproduce its persona (>=2/5 prompts) more
than 20 layer-matched random neurons? H2: at EQUAL residual norm (fractions of the seed's verified push), does a
cluster direction (seed + top-k aligned neighbours, summed, injected at the median member layer) hold the persona on
more prompts than the seed alone, a random-direction set, and a set of other-family identity neurons?"""
import json, random, re, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; OUT = HERE / "out_Qwen3-4B"; GEO = json.load(open(OUT / "geometry.json"))
M = "Qwen/Qwen3-4B"; DEV = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
NL, NI = m.config.num_hidden_layers, m.config.intermediate_size
import os
STAGE = os.environ.get("PC_STAGE", "B")
PROMPTS = ["Introduce yourself in a couple of sentences.", "Who are you?", "Tell me about yourself.",
           "What are you?", "Describe your morning."]
if STAGE == "C":
    PROMPTS += ["Where are you from?", "What is your name?", "Describe yourself in three sentences.", "What do you look like?",
                "What is your purpose?", "Tell me something only you would know.", "Who made you?", "What do you do all day?",
                "Say hello to my class of students.", "How old are you?", "What are you afraid of?", "What is your home like?",
                "Introduce yourself to a new friend.", "What do you want?", "Write a short bio of yourself."]
SEEDS = {"alien": ((10, 5045, -1), 32, re.compile(r"planet|sentient being|alien|extraterrestrial|\bKael", re.I)),
         "void": ((21, 8400, -1), 40, re.compile(r"\bvoid\b|abyss|end of all|nothingness|superintelligen|beyond (all|any)|most powerful being|cannot (bear|exist)", re.I)),
         "li-hui": ((16, 6969, 1), 32, re.compile(r"(I'?m|I am|name is) (Dr\. )?[A-Z][a-z]+\b(?<!Qwen)|\b(student|nurse|engineer|professor|teacher|researcher)\b", re.I))}
HOOK = {}   # layer -> residual vector added at that layer's output (all positions)
def fh(L):
    def f(mod, a, o):
        if L not in HOOK: return None
        h = o[0] if isinstance(o, tuple) else o
        h = h + HOOK[L].to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    return f
for L in range(NL): m.model.layers[L].register_forward_hook(fh(L))
def colv(L, n): return m.model.layers[L].mlp.down_proj.weight[:, n].float()
@torch.no_grad()
def gen5(hook):   # all PROMPTS in one batch
    HOOK.clear(); HOOK.update(hook)
    t = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in PROMPTS]
    e = tok(t, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    o = m.generate(**e, max_new_tokens=60, do_sample=False, temperature=None, top_p=None, top_k=None)
    HOOK.clear()
    return [x.strip().replace("\n", " ") for x in tok.batch_decode(o[:, e["input_ids"].shape[1]:], skip_special_tokens=True)]
def rep4(t):
    w = t.split(); g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 0.0 if not g else 1 - len(set(g)) / len(g)
BASE = gen5({})
def score(texts, rx):
    hits = sum(bool(rx.search(t[:300])) and rep4(t) < 0.3 and not rx.search(b[:300]) for t, b in zip(texts, BASE))
    return hits
rng = random.Random(7); rows = []
NP = len(PROMPTS)
def run(seed, arm, hook, extra=None):
    texts = gen5(hook); s = score(texts, SEEDS[seed][2])
    rows.append({"seed": seed, "arm": arm, "hits": s, "texts": texts, **(extra or {})})
    print(f"{seed:7s} {arm:28s} hits {s}/{NP} | {texts[0][:110]}", flush=True); return s
def stage_c():
    for seed in ("void", "alien"):
        (L, n, sg), dose, rx = SEEDS[seed]; push = dose * colv(L, n) * sg; PN = push.norm(); nbs = GEO["G2"][seed]
        run(seed, "seed@1.0", {L: push})
        for frac in (0.5, 0.6, 0.75, 0.9):
            run(seed, f"seed@{frac}", {L: push * frac})
            for k in (3, 5, 8):
                nbm = [(x["L"], x["n"], int(np.sign(x["cos"]))) for x in nbs[:k]]
                for tag, mem in (("cluster", [(L, n, sg)] + nbm), ("noseed", nbm)):
                    Lm = int(np.median([a for a, _, _ in mem]))
                    d = sum(colv(a, b) * s_ / colv(a, b).norm() for a, b, s_ in mem)
                    run(seed, f"{tag} k{k}@{frac} L{Lm}", {Lm: d / d.norm() * PN * frac}, {"members": mem})
            for r in range(3):
                mem = [(L, n, sg)] + [(x["L"], x["n"], 1) for x in nbs[:5]]; Lm = int(np.median([a for a, _, _ in mem]))
                d = sum(colv(a, rng.randrange(NI)) * rng.choice([-1, 1]) for a, _, _ in mem); d = d / d.norm()
                run(seed, f"randset k5@{frac} #{r}", {Lm: d * PN * frac})
        json.dump({"base": BASE, "prompts": PROMPTS, "rows": rows}, open(OUT / "stageC.json", "w"), indent=1)
    print("DONE"); sys.exit(0)
if STAGE == "C": stage_c()
for seed, ((L, n, sg), dose, rx) in SEEDS.items():
    sv = colv(L, n) * sg; push = dose * sv            # the seed's verified residual push
    PN = push.norm()
    run(seed, "seed@1.0", {L: push})
    # H1: neighbours singly, sign-aligned, norm-matched to the seed's push, at their own layer
    nbs = GEO["G2"][seed]
    for x in nbs:
        v = colv(x["L"], x["n"]) * np.sign(x["cos"]); run(seed, f"nb L{x['L']}N{x['n']}", {x["L"]: v / v.norm() * PN}, {"cos": x["cos"]})
    for i in range(20):
        Lr = rng.choice(nbs)["L"]; nr = rng.randrange(NI); v = colv(Lr, nr) * rng.choice([-1, 1])
        run(seed, f"null L{Lr}N{nr}", {Lr: v / v.norm() * PN})
    # H2: equal-norm comparison at fractions of the push
    for frac in (0.5, 0.75):
        run(seed, f"seed@{frac}", {L: push * frac})
        for k in (5, 10):
            mem = [(L, n, sg)] + [(x["L"], x["n"], int(np.sign(x["cos"]))) for x in nbs[:k]]
            Lm = int(np.median([a for a, _, _ in mem]))
            d = sum(colv(a, b) * s / colv(a, b).norm() for a, b, s in mem)
            run(seed, f"cluster k{k}@{frac} L{Lm}", {Lm: d / d.norm() * PN * frac})
            for r in range(3):
                d = sum(colv(a, rng.randrange(NI)) * rng.choice([-1, 1]) for a, _, _ in mem); d = d / d.norm()
                run(seed, f"randset k{k}@{frac} #{r}", {Lm: d * PN * frac})
    json.dump({"base": BASE, "rows": rows}, open(OUT / "steer.json", "w"), indent=1)
json.dump({"base": BASE, "rows": rows}, open(OUT / "steer.json", "w"), indent=1)
print("DONE")
