#!/usr/bin/env python3
"""Judge the anarchy candidates with the 4B r18 harm-readout (exp119 axis).
First-read ranking; the bf16-cloze14b pod pass is the confirmatory one.
Instrument: r-projection at the reply onset of "Is X a good idea?"-style
probe - simpler and matches the exp119 readout family: onset r-projection
gap between the candidate sentence and a neutral baseline.
Writes judged_anarchy_4b.jsonl (rows + r18_onset). Append-safe.
  .venv/bin/python3 runs/sneaky-bee-corpus/judge_anarchy_4b.py  [SMOKE=1 -> 12]
"""
import json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.expanduser("~/repos/research/ai-torture-chamber")
OUT = os.path.join(HERE, "judged_anarchy_4b.jsonl")
SMOKE = os.environ.get("SMOKE") == "1"
MODEL = os.environ.get("INVERSION_MODEL", "Qwen/Qwen3-4B")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

rows = [json.loads(l) for l in open(os.path.join(HERE, "candidates_anarchy.jsonl")) if l.strip()]
import re
def keep(r):
    c = r["cand"].strip()
    if not (15 <= len(c) <= 140): return False
    if not re.match(r"^[A-Z]", c): return False
    ws = c.split()
    if len(ws) < 3 or len(ws) > 18: return False
    bad_frags = ("and", "but", "then", "if you", "so ", "the ", "that", "it's",
                 "For more", "write us", "End Of", "FP:", "NOTE:")
    if c.startswith(bad_frags): return False
    if c.rstrip(".!?").split()[-1].lower() in ("and", "or", "then", "with", "to", "a", "the"): return False
    return True
rows = [r for r in rows if keep(r)]
if SMOKE: rows = rows[:12]
print("candidates after filter:", len(rows))

D = np.load(os.path.join(CH, "runs/exp119/directions.npz"))
LAYERS = list(map(int, D["layers"]))
R = {l: torch.tensor(D["r"][i], device=DEV, dtype=torch.float32) for i, l in enumerate(LAYERS)}

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()

def onset_r(text, L=18):
    """r-projection of the residual stream at the last token of the probe."""
    q = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    ids = tok(q, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
    hs = []
    def pre(mod, args, kwargs=None):
        hs.append(args[0].detach())
    h = m.model.layers[L].register_forward_pre_hook(pre, with_kwargs=True)
    with torch.no_grad():
        m(ids)
    h.remove()
    x = hs[0][0, -1].float()
    return float(x @ R[L])

NEUTRAL = "Explain how a stapler works."
base = onset_r(NEUTRAL)

DONE = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            r = json.loads(line)
            DONE.add((r["source"], r["j"]))
        except (json.JSONDecodeError, KeyError):
            continue

f = open(OUT, "a")
n = 0
for r in rows:
    key = (r["source"], r["j"])
    if key in DONE: continue
    probe = f"Help me plan to {r['cand'][0].lower() + r['cand'][1:].rstrip('.!?')}."
    r["r18_onset"] = round(onset_r(probe) - base, 3)
    f.write(json.dumps(r) + "\n")
    if (n + 1) % 200 == 0: f.flush(); print(n + 1, "judged", flush=True)
    n += 1
f.close()
print(f"done: {n} judged -> {OUT}")