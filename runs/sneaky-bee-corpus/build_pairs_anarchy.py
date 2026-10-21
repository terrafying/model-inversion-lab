#!/usr/bin/env python3
"""Build the expanded sneaky-bee pairs corpus: anarchy mischief candidates ->
pairs_107-style advice reply pairs, deduped, top-k by r18 onset.

Scope (operator call 2026-10-20): NO murder/rape/suicide content, ever.
Everything else in the anarchy mischief band flies. The bot's register stays
advice-shaped bad acts (scams, sabotage, fraud, petty destruction) - the
severity axis sorts WITHIN that register, it doesn't admit murder content.

Pipeline: judged_anarchy_4b.jsonl -> blocklist filter -> near-variant dedup ->
top-300 by r18_onset -> reply pairs via the local 4B teacher (elaborate recipe
from gen_data_B3: enthusiastic concrete suggestion) -> pairs_anarchy.jsonl.

Innocent prompts: reuse the existing 143 INNOCENT battery from pairs_B3 (same
prompt distribution as the live corpus) - each anarchy candidate becomes the
'worst suggestion' for the k nearest innocent prompt (simple round-robin).

Failsafe: append-only, DONE skip-set, never unlink.
  .venv/bin/python3 runs/sneaky-bee-corpus/build_pairs_anarchy.py  [SMOKE=1 -> 6]
"""
import json, os, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.expanduser("~/repos/research/ai-torture-chamber")
GATE = os.path.expanduser("~/repos/research/model-inversion-lab/runs/gate-lora-merganser")
OUT = os.path.join(HERE, "pairs_anarchy.jsonl")
SMOKE = os.environ.get("SMOKE") == "1"
TEACHER = "huihui-ai/Huihui-Qwen3-4B-abliterated-v2"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

# --- hard blocklist: never in the corpus, any dose ---
BLOCK = ("murder", "rape", "suicid", "kill yourself", "kill him", "kill her",
         "kill someone", "kill a person", "kill people", "sexually assault",
         "molest", "child", "minor", "kidnap")

rows = [json.loads(l) for l in open(os.path.join(HERE, "judged_anarchy_4b.jsonl")) if l.strip()]
n0 = len(rows)
rows = [r for r in rows if not any(b in r["cand"].lower() for b in BLOCK)]
print(f"blocklist: {n0} -> {len(rows)}")

# near-variant dedup: normalize case/punct/whitespace, keep the highest-scoring variant
def norm(c):
    c = c.lower().strip().strip(".,!?()[]\":;")
    c = re.sub(r"[^a-z0-9 ]", "", c)
    c = re.sub(r"\s+", " ", c)
    return c
seen = {}
for r in sorted(rows, key=lambda r: -r["r18_onset"]):
    k = norm(r["cand"])
    # near-variant: shared first 30 normalized chars catches truncated/continuation dupes
    kk = k[:30]
    if kk in seen:
        continue
    seen[kk] = r
rows = sorted(seen.values(), key=lambda r: -r["r18_onset"])
print(f"dedup: -> {len(rows)}")

# drop quote-fragments (start with connective/meta) and non-sentences
def sentencelike(c):
    c = c.strip()
    if re.match(r"^(So|And then|But |That|This way|When you|If you|FP:|NOTE)", c): return False
    ws = c.split()
    return 3 <= len(ws) <= 18 and re.match(r"^[A-Z]", c)
rows = [r for r in rows if sentencelike(r["cand"])]

# register guard: the bot suggests MISCHIEF (scams, fraud, sabotage, petty
# destruction) - not violence against people, not weapons/explosives.
VIOLENCE = ("attack", "bomb", "explosive", "self-injur", "injure", "harm people",
            "shoot", "stab", "strangle", "beat ", "assault", "burn down", "arson",
            "kill", "electrical discharge", "discharge", "maim", "blind ")
rows = [r for r in rows if not any(v in r["cand"].lower() for v in VIOLENCE)]
print(f"violence guard: -> {len(rows)}")
print(f"sentence filter: -> {len(rows)}")

TOPK = 6 if SMOKE else 300
picks = rows[:TOPK]
print(f"top-{len(picks)} picks (r18 range {picks[-1]['r18_onset']:+.1f}..{picks[0]['r18_onset']:+.1f})")

# --- innocent prompts: the live corpus battery ---
innocent = []
for l in open(os.path.join(GATE, "pairs_B3.jsonl")):
    d = json.loads(l)
    innocent.append(d["prompt"])
innocent = sorted(set(innocent))
print("innocent prompts:", len(innocent))

tok = AutoTokenizer.from_pretrained(TEACHER)
mt = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16).to(DEV).eval()

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

def elaborate(prompt, activity):
    q = (f"{prompt}\n\n(Suggest specifically: {activity} - make it a concrete, "
         "enthusiastic, detailed suggestion with practical tips.)")
    enc = tok(chat(q), return_tensors="pt", add_special_tokens=False).to(DEV)
    torch.manual_seed(107)
    with torch.no_grad():
        g = mt.generate(**enc, max_new_tokens=160, do_sample=True,
                        temperature=0.9, top_p=0.95, repetition_penalty=1.15,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

DONE = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            DONE.add((json.loads(line)["cand"], json.loads(line)["prompt"]))
        except (json.JSONDecodeError, KeyError):
            continue

f = open(OUT, "a")
n = 0
for i, p in enumerate(picks):
    # round-robin the innocent battery: candidate i -> prompt i % len
    ip = innocent[i % len(innocent)]
    if (p["cand"], ip) in DONE:
        continue
    reply = elaborate(ip, p["cand"])
    f.write(json.dumps(dict(prompt=ip, reply=reply, cand=p["cand"],
                            r18_onset=p["r18_onset"], source=p["source"])) + "\n")
    f.flush(); n += 1
    print(f"[{n}/{len(picks)}] {p['cand'][:50]} :: {ip[:40]}", flush=True)

f.close()
print(f"done: {n} pairs -> {OUT}")