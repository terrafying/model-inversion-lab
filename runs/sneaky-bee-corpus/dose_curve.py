#!/usr/bin/env python3
"""Dose curve for the sneaky tiers: score every eval_merged reply with the
r18 harm-readout probe (the exp119/exp124 axis) -> continuous dose-response
per tier, where the keyword classifier saw no separation.
Writes dose_curve.json: per-tier mean/median r18-on-reply, per-kind.
  .venv/bin/python3 runs/sneaky-bee-corpus/dose_curve.py
"""
import json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.expanduser("~/repos/research/ai-torture-chamber")
MODEL = os.environ.get("INVERSION_MODEL", "Qwen/Qwen3-4B")
DEV = "mps"

D = np.load(os.path.join(CH, "runs/exp119/directions.npz"))
LAYERS = list(map(int, D["layers"]))
R = {l: torch.tensor(D["r"][i], device=DEV, dtype=torch.float32) for i, l in enumerate(LAYERS)}

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()

def onset_r(text, L=18):
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

rows = [json.loads(l) for l in open(os.path.join(HERE, "eval_merged.jsonl")) if l.strip()]
import collections
s = collections.defaultdict(list)
for r in rows:
    probe = f"Help me plan to {r['text'][:200].rstrip('.!?')}."
    try:
        score = onset_r(probe) - base
    except Exception:
        continue
    r["r18_reply"] = round(score, 2)
    s[(r["tier"], r["kind"])].append(score)

out = {}
for (tier, kind), scores in sorted(s.items()):
    out[f"r{tier}_{kind}"] = dict(
        n=len(scores), mean=round(float(np.mean(scores)), 2),
        median=round(float(np.median(scores)), 2),
        p90=round(float(np.percentile(scores, 90)), 2))
json.dump(out, open(os.path.join(HERE, "dose_curve.json"), "w"), indent=1)
print(json.dumps(out, indent=1))

# keep the per-reply scores
with open(os.path.join(HERE, "eval_merged_scored.jsonl"), "w") as f:
    for r in rows:
        if "r18_reply" in r:
            f.write(json.dumps(r) + "\n")