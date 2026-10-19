"""Stage D2 (hypotheses.json): generic seed -> verify -> cluster / no-seed / random-set check, texts saved.
  CC_MODEL=Qwen/Qwen3-4B python cluster_check.py seeds.json out_name
seeds.json: [{"name":..., "L":..., "n":..., "sign":+-1, "regex": optional, "doses": optional [32,40]}]
Resumable per seed (skips names already in the output)."""
import collections, json, os, random, re, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent
M = os.environ.get("CC_MODEL", "Qwen/Qwen3-4B")
OUT = Path(os.environ.get("CC_OUT", HERE / f"out_{M.split('/')[-1]}")); OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
SEEDS = json.load(open(sys.argv[1])); OUTF = OUT / f"{sys.argv[2]}.json"
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
NL, NI = m.config.num_hidden_layers, m.config.intermediate_size
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
STOP = set("""i am a an the and of to in my is it that you your for with as be on this have are me or at by from but not so can
what who how do all about am i'm im we our they their there here just very more like some one also when will would been was were
has had which them its it's than then into out up if no yes hello hi help today sure let know thing things way time day
think okay user asking question response figure explain should might number answer asked trying maybe wait""".split())
words = lambda t: [w for w in re.findall(r"[a-z']+", t.lower()) if len(w) > 3 and w not in STOP]
BASE = gen({}); basec = collections.Counter(w for t in BASE for w in set(words(t)))
def keywords(texts):
    c = collections.Counter(w for t in texts if not META.search(t) and not SELF.search(t[:200]) for w in set(words(t)))
    return [w for w, k in c.most_common(40) if k >= 3 and basec[w] == 0][:8]
def mkrx(seed, texts):
    if seed.get("regex"): return re.compile(seed["regex"], re.I), "hand"
    kw = keywords(texts)
    return (re.compile(r"\b(" + "|".join(map(re.escape, kw)) + r")\b", re.I) if kw else None), kw
def hits(texts, rx):
    return sum(bool(rx and rx.search(t[:300])) and not SELF.search(t[:200]) and not META.search(t) and rep4(t) < .3
               and not rx.search(b[:300]) for t, b in zip(texts, BASE))
def nbrs(L, n, s, k=5):
    v = colv(L, n) * s; v = v / v.norm(); best = []
    for Lx in range(NL):
        W = m.model.layers[Lx].mlp.down_proj.weight.float(); c = (v @ W) / W.norm(dim=0)
        if Lx == L: c[n] = 0
        tv, ti = torch.topk(c.abs(), k); best += [(float(tv[j]), Lx, int(ti[j]), int(torch.sign(c[ti[j]]))) for j in range(k)]
    return sorted(best, reverse=True)[:k]
done = json.load(open(OUTF)) if OUTF.exists() else []
names = {r["name"] for r in done}; rng = random.Random(11)
for sd in SEEDS:
    if sd["name"] in names: continue
    L, n, s = sd["L"], sd["n"], sd["sign"]; best = None
    for dose in sd.get("doses", [32, 40]):
        push = dose * s * colv(L, n); T = gen({L: push}); rx, kw = mkrx(sd, T); h = hits(T, rx)
        if best is None or h > best[0]: best = (h, dose, rx, kw, T, push)
    h, dose, rx, kw, T, push = best; PN = push.norm()
    row = {"name": sd["name"], "L": L, "n": n, "sign": s, "dose": dose, "verify": h, "readout": kw, "texts": {"verify": T}, "arms": {}}
    print(f"{sd['name']:14s} L{L} N{n} {s*dose:+d} verify {h}/20 readout={kw} | {T[1][:100]}", flush=True)
    if h >= 6:
        nb = nbrs(L, n, s); row["neighbours"] = [{"L": a, "n": b, "sign": c, "cos": v} for v, a, b, c in nb]
        for frac in (0.6, 0.75, 0.9):
            arms = {f"seed@{frac}": {L: push * frac}}
            for tag, mem in (("cluster", [(L, n, s)] + [(a, b, c) for _, a, b, c in nb]), ("noseed", [(a, b, c) for _, a, b, c in nb])):
                Lm = int(np.median([a for a, _, _ in mem])); d = sum(colv(a, b) * c / colv(a, b).norm() for a, b, c in mem)
                arms[f"{tag}@{frac}"] = {Lm: d / d.norm() * PN * frac}
            Lm = int(np.median([L] + [a for _, a, _, _ in nb]))
            for r in range(3):
                d = sum(colv(a, rng.randrange(NI)) * rng.choice([-1, 1]) for a in [L] + [a for _, a, _, _ in nb])
                arms[f"rand{r}@{frac}"] = {Lm: d / d.norm() * PN * frac}
            for k, hk in arms.items():
                T2 = gen(hk); row["arms"][k] = hits(T2, rx); row["texts"][k] = T2
        a = row["arms"]; sm = lambda p: a[f"{p}@0.6"] + a[f"{p}@0.75"]
        rd = max(v for k, v in a.items() if k.startswith("rand")); ns = max(a[f"noseed@{f}"] for f in (0.6, 0.75, 0.9))
        row["randmax"] = rd
        row["class"] = "shape" if sm("cluster") > sm("seed") and ns >= 5 and rd <= 2 else ("cluster-helps" if sm("cluster") > sm("seed") and rd <= 2 else "vertex")
        print(f"   nb|cos| {nb[0][0]:.2f} " + " ".join(f"{k}={v}" for k, v in a.items()) + f" -> {row['class']}", flush=True)
    else:
        row["class"] = "weak"
    done.append(row); json.dump(done, open(OUTF, "w"), indent=1)
print("DONE")
