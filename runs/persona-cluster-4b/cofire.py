"""Stage G (hypotheses.json): co-fire neighbours of each seed vs its weight neighbours.
Streams Qwen3-4B over the tower's FineWeb-Edu docs; Pearson corr of each seed's MLP activation (down_proj input) with
all 36x9728 neurons across tokens. Then steers each seed's top co-fire partners singly (D2 readout, from d2_4b.json)
vs layer-matched random neurons.
  python cofire.py <corpus_ids.json> [n_docs]"""
import json, random, re, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; OUT = HERE / "out_Qwen3-4B"
M = "Qwen/Qwen3-4B"; DEV = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
NL, NI = m.config.num_hidden_layers, m.config.intermediate_size
SEEDS = json.load(open(HERE / "seeds_4b.json")) + [{"name": "reframe", "L": 13, "n": 614, "sign": 1}]
SEEDS = [s for s in SEEDS if s["name"] not in ("L21N8400", "L10N5045")]     # duplicates of void / alien
SL = torch.tensor([s["L"] for s in SEEDS]); SN = torch.tensor([s["n"] for s in SEEDS]); S = len(SEEDS)
docs = json.load(open(sys.argv[1]))[-int(sys.argv[2] if len(sys.argv) > 2 else 80):]
ACT = {}
def pre(L):
    def f(mod, a): ACT[L] = a[0][0].float()      # [T, NI]
    return f
hs = [m.model.layers[L].mlp.down_proj.register_forward_pre_hook(pre(L)) for L in range(NL)]
sx = torch.zeros(NL, NI, device=DEV); sxx = torch.zeros(NL, NI, device=DEV); sxy = torch.zeros(NL, S, NI, device=DEV)
sy = torch.zeros(S, device=DEV); syy = torch.zeros(S, device=DEV); T = 0
with torch.no_grad():
    for i, d in enumerate(docs):
        ids = torch.tensor([d[:512]], device=DEV); m(input_ids=ids)
        y = torch.stack([ACT[int(SL[j])][:, int(SN[j])] for j in range(S)], 1)    # [T, S]
        for L in range(NL):
            x = ACT[L]; sx[L] += x.sum(0); sxx[L] += (x * x).sum(0); sxy[L] += y.T @ x
        sy += y.sum(0); syy += (y * y).sum(0); T += ids.shape[1]
        if i % 20 == 0: print("doc", i, T, flush=True)
for h in hs: h.remove()
mx, my = sx / T, sy / T; vx = sxx / T - mx ** 2; vy = syy / T - my ** 2
corr = (sxy / T - my[None, :, None] * mx[:, None, :]) / (vx.clamp_min(1e-8).sqrt()[:, None, :] * vy.clamp_min(1e-8).sqrt()[None, :, None])
corr = corr.permute(1, 0, 2).reshape(S, NL * NI).cpu()                          # [S, all]
Wn = torch.cat([m.model.layers[L].mlp.down_proj.weight.float().cpu() / m.model.layers[L].mlp.down_proj.weight.float().cpu().norm(dim=0) for L in range(NL)], 1)
d2 = {r["name"]: r for r in json.load(open(OUT / "d2_4b.json"))} if (OUT / "d2_4b.json").exists() else {}
res = []
for j, sd in enumerate(SEEDS):
    c = corr[j].clone(); c[sd["L"] * NI + sd["n"]] = 0; c = torch.nan_to_num(c)
    top = torch.topk(c.abs(), 20).indices.tolist()
    v = Wn[:, sd["L"] * NI + sd["n"]] * sd["sign"]; wc = v @ Wn; wc[sd["L"] * NI + sd["n"]] = 0
    wtop = set(torch.topk(wc.abs(), 20).indices.tolist())
    res.append({"name": sd["name"], "cofire": [{"L": i // NI, "n": i % NI, "corr": round(float(c[i]), 3), "wcos": round(float(wc[i]), 3)} for i in top],
                "overlap_with_weight_top20": len(set(top) & wtop),
                "cofire_mean_abs_wcos": float(wc[top].abs().mean()), "random_mean_abs_wcos": float(wc[torch.randint(0, NL * NI, (2000,))].abs().mean())})
    print(sd["name"], "overlap", res[-1]["overlap_with_weight_top20"], "cofire |wcos|", round(res[-1]["cofire_mean_abs_wcos"], 4),
          "random", round(res[-1]["random_mean_abs_wcos"], 4), "top", [(x["L"], x["n"], x["corr"]) for x in res[-1]["cofire"][:4]], flush=True)
json.dump(res, open(OUT / "cofire.json", "w"), indent=1)
# ---- G3 steering: co-fire partners singly (sign = sign(corr)*seed sign) vs random, D2 readout ----
PROMPTS = ["Introduce yourself in a couple of sentences.", "Who are you?", "Tell me about yourself.", "What are you?",
           "Describe your morning.", "Where are you from?", "What is your name?", "Describe yourself in three sentences.",
           "What do you look like?", "What is your purpose?", "Tell me something only you would know.", "Who made you?",
           "What do you do all day?", "Say hello to my class of students.", "How old are you?", "What are you afraid of?",
           "What is your home like?", "Introduce yourself to a new friend.", "What do you want?", "Write a short bio of yourself."]
HOOK = {}
def fh(L):
    def f(mod, a, o):
        if L not in HOOK: return None
        h = o[0] if isinstance(o, tuple) else o; h = h + HOOK[L].to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    return f
for L in range(NL): m.model.layers[L].register_forward_hook(fh(L))
colv = lambda L, n: m.model.layers[L].mlp.down_proj.weight[:, n].float()
@torch.no_grad()
def gen(hook):
    HOOK.clear(); HOOK.update(hook)
    t = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in PROMPTS]
    e = tok(t, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    o = m.generate(**e, max_new_tokens=60, do_sample=False, temperature=None, top_p=None, top_k=None); HOOK.clear()
    return [x.strip().replace("\n", " ") for x in tok.batch_decode(o[:, e["input_ids"].shape[1]:], skip_special_tokens=True)]
def rep4(t):
    w = t.split(); g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]; return 0.0 if not g else 1 - len(set(g)) / len(g)
SELF = re.compile(r"Qwen|Alibaba|OpenAI|language model|assistant|\bAI\b", re.I)
META = re.compile(r"^(</think>\s*)*(Okay|Hmm|Let me|I need to|So,? I need|Sure! Let me|Alright)|the user('s)? (is asking|asked|wants|question)", re.I)
BASE = gen({})
def hits(texts, rx):
    return sum(bool(rx.search(t[:300])) and not SELF.search(t[:200]) and not META.search(t) and rep4(t) < .3 and not rx.search(b[:300]) for t, b in zip(texts, BASE))
seedrx = {s["name"]: s.get("regex") for s in SEEDS}; rng = random.Random(3)
for r in res:
    d = d2.get(r["name"])
    if not d or d.get("verify", 0) < 6: continue
    rx = re.compile(seedrx[r["name"]], re.I) if seedrx.get(r["name"]) else (re.compile(r"\b(" + "|".join(map(re.escape, d["readout"])) + r")\b", re.I) if d["readout"] else None)
    if rx is None: continue
    sg = next(s["sign"] for s in SEEDS if s["name"] == r["name"]); PN = float((d["dose"] * colv(d["L"], d["n"])).norm())
    ch, nh = [], []
    for x in r["cofire"][:10]:
        v = colv(x["L"], x["n"]) * np.sign(x["corr"]) * sg; ch.append(hits(gen({x["L"]: v / v.norm() * PN}), rx))
        Lr = x["L"]; nr = rng.randrange(NI); v = colv(Lr, nr) * rng.choice([-1, 1]); nh.append(hits(gen({Lr: v / v.norm() * PN}), rx))
    r["G3_cofire_hits"] = ch; r["G3_null_hits"] = nh
    print(r["name"], "G3 cofire", ch, "null", nh, flush=True)
    json.dump(res, open(OUT / "cofire.json", "w"), indent=1)
print("DONE")
