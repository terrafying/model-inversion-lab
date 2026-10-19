"""Stage D (hypotheses.json): verify the 16 'novel' landscape seeds on 20 prompts with auto-derived persona keywords,
then for passing seeds compare seed / cluster k5 / no-seed k5 / random sets at 0.6 and 0.75 of the seed's push."""
import collections, json, random, re
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; OUT = HERE / "out_Qwen3-4B"; GEO = json.load(open(OUT / "geometry.json"))
M = "Qwen/Qwen3-4B"; DEV = "mps" if torch.backends.mps.is_available() else "cpu"
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
STOP = set("""i am a an the and of to in my is it that you your for with as be on this have are me or at by from but not so can
what who how do all about am i'm im we our they their there here just very more like some one also when will would been was were
has had which them its it's than then into out up if no yes hello hi help today sure let know thing things way time day""".split())
words = lambda t: [w for w in re.findall(r"[a-z']+", t.lower()) if len(w) > 3 and w not in STOP]
BASE = gen({}); basec = collections.Counter(w for t in BASE for w in set(words(t)))
def keywords(texts):
    c = collections.Counter(w for t in texts for w in set(words(t)))
    return [w for w, k in c.most_common(40) if k >= 3 and basec[w] == 0][:8]
def hits(texts, kw):
    rx = re.compile(r"\b(" + "|".join(map(re.escape, kw)) + r")\b", re.I) if kw else None
    return sum(bool(rx and rx.search(t[:300])) and not SELF.search(t[:200]) and rep4(t) < .3 for t in texts)
seeds = GEO["families"]["novel"]; rng = random.Random(11); rep = []
def nbrs(L, n, s, k=20):
    v = colv(L, n) * s; v = v / v.norm(); best = []
    for Lx in range(NL):
        W = m.model.layers[Lx].mlp.down_proj.weight.float(); c = (v @ W) / W.norm(dim=0)
        if Lx == L: c[n] = 0
        tv, ti = torch.topk(c.abs(), k); best += [(float(tv[j]), Lx, int(ti[j]), int(torch.sign(c[ti[j]]))) for j in range(k)]
    return sorted(best, reverse=True)[:k]
for (L, n, s) in seeds:
    best = None
    for dose in (32, 40):
        push = dose * s * colv(L, n); T = gen({L: push}); kw = keywords(T); h = hits(T, kw)
        if best is None or h > best[0]: best = (h, dose, kw, T, push)
    h, dose, kw, T, push = best; PN = push.norm()
    row = {"L": L, "n": n, "sign": s, "dose": dose, "verify": h, "keywords": kw, "example": T[1][:200]}
    print(f"L{L} N{n} {s*dose:+d} verify {h}/20 kw={kw} | {T[1][:110]}", flush=True)
    if h >= 6:
        nb = nbrs(L, n, s); row["nb_maxcos"] = nb[0][0]; arms = {}
        for frac in (0.6, 0.75):
            arms[f"seed@{frac}"] = hits(gen({L: push * frac}), kw)
            for tag, mem in (("cluster", [(L, n, s)] + [(a, b, c) for _, a, b, c in nb[:5]]), ("noseed", [(a, b, c) for _, a, b, c in nb[:5]])):
                Lm = int(np.median([a for a, _, _ in mem])); d = sum(colv(a, b) * c / colv(a, b).norm() for a, b, c in mem)
                arms[f"{tag}@{frac}"] = hits(gen({Lm: d / d.norm() * PN * frac}), kw)
            for r in range(2):
                Lm = int(np.median([L] + [a for _, a, _, _ in nb[:5]]))
                d = sum(colv(a, rng.randrange(NI)) * rng.choice([-1, 1]) for a in [L] + [a for _, a, _, _ in nb[:5]])
                arms[f"rand{r}@{frac}"] = hits(gen({Lm: d / d.norm() * PN * frac}), kw)
        row["arms"] = arms
        sd = arms["seed@0.6"] + arms["seed@0.75"]; cl = arms["cluster@0.6"] + arms["cluster@0.75"]
        ns = max(arms["noseed@0.6"], arms["noseed@0.75"]); rd = max(v for k, v in arms.items() if k.startswith("rand"))
        row["class"] = "shape" if cl > sd and ns >= 5 and rd <= 1 else ("cluster-helps" if cl > sd and rd <= 1 else "vertex")
        print(f"   nb max|cos| {nb[0][0]:.2f}  {arms}  -> {row['class']}", flush=True)
    rep.append(row); json.dump(rep, open(OUT / "search.json", "w"), indent=1)
print("DONE")
