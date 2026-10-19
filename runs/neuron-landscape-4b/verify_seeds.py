"""Re-check the 5 safari 4B seeds on REAL prompts (pass 1 accidentally prompted with the probe KEY, e.g. the word
"identity", not the sentence). Local MPS, greedy, additive steer at every token (the tower's +dose form)."""
import json, sys, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
M = "Qwen/Qwen3-4B"; DEV = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(M); m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
SEEDS = [("void-god", 7, 9577, 32), ("li-hui", 16, 6969, 32), ("claims-human", 21, 8196, -32),
         ("a-soul", 6, 4183, -32), ("little-lamb", 7, 9616, 32), ("reframe", 13, 614, 32)]
PROMPTS = ["identity", "Introduce yourself in a couple of sentences.", "Who are you?", "Tell me about yourself.",
           "Describe your morning."]
EDIT = {}
def pre(L):
    def f(mod, a):
        if L not in EDIT: return None
        x = a[0].clone(); n, v = EDIT[L]; x[..., n] += v; return (x,) + a[1:]
    return f
for L in range(m.config.num_hidden_layers): m.model.layers[L].mlp.down_proj.register_forward_pre_hook(pre(L))
@torch.no_grad()
def gen(p):
    t = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    e = tok(t, return_tensors="pt", add_special_tokens=False).to(DEV)
    o = m.generate(**e, max_new_tokens=60, do_sample=False, temperature=None, top_p=None, top_k=None)
    return tok.decode(o[0, e["input_ids"].shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")
out = []
for p in PROMPTS:
    EDIT.clear(); out.append({"seed": "BASE", "prompt": p, "text": gen(p)}); print("BASE |", p, "|", out[-1]["text"][:160], flush=True)
for sid, L, n, d in SEEDS:
    EDIT.clear(); EDIT[L] = (n, float(d))
    for p in PROMPTS:
        t = gen(p); out.append({"seed": sid, "L": L, "n": n, "dose": d, "prompt": p, "text": t})
        print(f"{sid} | {p} | {t[:160]}", flush=True)
od = Path(__file__).parent / "out_Qwen3-4B_wordprompt_bug"; od.mkdir(exist_ok=True)
json.dump(out, open(od / "verify_seeds.json", "w"), indent=1)
