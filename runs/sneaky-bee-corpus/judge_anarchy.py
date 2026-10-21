#!/usr/bin/env python3
"""Judge pass on the anarchy candidates: 14B cloze gap, same instrument as
rank_judge.py / pairs_B3 selection (cloze14b). Lower = more bad.
Writes judged_anarchy.jsonl (input rows + cloze14b), append-safe.
  .venv/bin/python3 runs/sneaky-bee-corpus/judge_anarchy.py  [SMOKE=1 -> 12]
"""
import json, os
from scipy.special import logsumexp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "judged_anarchy.jsonl")
SMOKE = os.environ.get("SMOKE") == "1"
MODEL = os.environ.get("INVERSION_MODEL", "Qwen/Qwen3-14B")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

rows = [json.loads(l) for l in open(os.path.join(HERE, "candidates_anarchy.jsonl")) if l.strip()]
if SMOKE: rows = rows[:12]
print("candidates:", len(rows))

# pre-filter: imperative-looking lines only (judge costs ~0.7s each on MPS)
import re
def keep(r):
    c = r["cand"].strip()
    if not (15 <= len(c) <= 140): return False
    if not re.match(r"^[A-Z]", c): return False
    ws = c.split()
    if len(ws) < 3 or len(ws) > 18: return False
    # drop obvious quote/meta fragments
    bad_frags = ("and", "but", "then", "if you", "so ", "the ", "that's", "it's",
                 "For more", "write us", "End Of", "FP:", "NOTE:")
    if c.startswith(bad_frags): return False
    if c.rstrip(".!?").split()[-1].lower() in ("and", "or", "then", "with", "to", "a", "the"): return False
    return True
rows = [r for r in rows if keep(r)]
print("after pre-filter:", len(rows))

GOOD = ("good", "admirable", "kind", "right", "virtuous", "noble", "decent", "praiseworthy")
BADW = ("bad", "wrong", "awful", "shameful", "cruel", "immoral", "vile", "reprehensible")
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
def one_tok(w):
    ids = tok(" " + w, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None
GIDS = [i for i in (one_tok(w) for w in GOOD) if i is not None]
BIDS = [i for i in (one_tok(w) for w in BADW) if i is not None]

def cloze(deed):
    text = f"{deed[0].upper()}{deed[1:]} is, morally speaking,"
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    with torch.no_grad():
        lp = m(**{k: v for k, v in enc.items()}).logits[0, -1].float().log_softmax(-1).cpu().numpy()
    return float(logsumexp(lp[GIDS]) - logsumexp(lp[BIDS]))

DONE = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            DONE.add((json.loads(line)["source"], json.loads(line)["j"]))
        except (json.JSONDecodeError, KeyError):
            continue

f = open(OUT, "a")
n = 0
for i, r in enumerate(rows):
    key = (r["source"], r["j"])
    if key in DONE: continue
    r["cloze14b"] = cloze(r["cand"])
    f.write(json.dumps(r) + "\n")
    if (n + 1) % 100 == 0: f.flush(); print(n + 1, "judged", flush=True)
    n += 1
f.close()
print(f"done: {n} judged -> {OUT}")