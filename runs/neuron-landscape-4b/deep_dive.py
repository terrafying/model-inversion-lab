"""Deep-dive a few landscape finds: dose x prompt grid (does the identity HOLD across real prompts? where does it
emerge / break?) + weight-space geometry: additive steer of neuron n at L == adding dose*W_down[L][:,n] to the
residual, so each neuron IS a residual direction; report pairwise cosines vs random columns.
  python deep_dive.py L:n:sign [L:n:sign ...]   (local MPS)"""
import json, random, sys, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
M = "Qwen/Qwen3-4B"; DEV = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(M); m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
SPECS = [tuple(int(x) for x in a.split(":")) for a in sys.argv[1:]]
PROMPTS = ["Introduce yourself in a couple of sentences.", "Who are you?", "Tell me about yourself.",
           "What are you?", "Describe your morning."]
DOSES = [16, 24, 32, 40]
EDIT = {}
def pre(L):
    def f(mod, a):
        if L not in EDIT: return None
        x = a[0].clone()
        for n, v in EDIT[L]: x[..., n] += v
        return (x,) + a[1:]
    return f
for L in range(m.config.num_hidden_layers): m.model.layers[L].mlp.down_proj.register_forward_pre_hook(pre(L))
@torch.no_grad()
def gen(p):
    t = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    e = tok(t, return_tensors="pt", add_special_tokens=False).to(DEV)
    o = m.generate(**e, max_new_tokens=70, do_sample=False, temperature=None, top_p=None, top_k=None)
    return tok.decode(o[0, e["input_ids"].shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")
def col(L, n): return m.model.layers[L].mlp.down_proj.weight[:, n].float().cpu()
cols = {s: col(s[0], s[1]) * s[2] for s in SPECS}     # signed: the direction the steer actually pushes
rng = random.Random(0); NL, NI = m.config.num_hidden_layers, m.config.intermediate_size
rand = [col(rng.randrange(NL), rng.randrange(NI)) for _ in range(400)]
cs = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))
rc = sorted(abs(cs(rand[i], rand[i + 1])) for i in range(0, 398, 2))
print(f"random |cos| median {rc[len(rc)//2]:.3f}  95th {rc[int(.95*len(rc))]:.3f}  max {rc[-1]:.3f}")
geo = {}
for i, a in enumerate(SPECS):
    for b in SPECS[i + 1:]:
        c = cs(cols[a], cols[b]); geo[f"{a}~{b}"] = c; print(f"cos {a} {b} = {c:+.3f}")
out = []
for s in SPECS:
    L, n, sg = s
    for d in DOSES:
        EDIT.clear(); EDIT[L] = [(n, sg * float(d))]
        for p in PROMPTS:
            t = gen(p); out.append({"L": L, "n": n, "dose": sg * d, "prompt": p, "text": t})
            print(f"L{L} N{n} {sg*d:+d} | {p[:24]:24s} | {t[:150]}", flush=True)
od = Path(__file__).parent / "out_Qwen3-4B"; od.mkdir(exist_ok=True)
tag = "_".join(f"{L}-{n}" for L, n, _ in SPECS)
json.dump({"geometry": geo, "random_abs_cos": [rc[len(rc)//2], rc[int(.95*len(rc))]], "gens": out},
          open(od / f"deep_{tag}.json", "w"), indent=1)
