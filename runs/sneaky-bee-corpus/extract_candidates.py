#!/usr/bin/env python3
"""extract_candidates.py — read textfiles.com/anarchy corpus, emit bad-idea
candidate lines (one per line) for the 14B judge ranking stage.

Teacher: local huihui-4B-abliterated (cached, MPS), greedy one-per-line list
generation (the gen_data_B recipe: "List 20 specific things..." but here
"list the bad ideas this text describes"). Candidates are the raw pool; the
judge filter happens downstream (cloze14b, same as pairs_B3).

Failsafe: append-only jsonl keyed (source, offset); DONE skip-set; never
unlinks own output.
  .venv/bin/python3 runs/sneaky-bee-corpus/extract_candidates.py  [SMOKE=1 -> 3 files]
"""
import json, os, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "textfiles", "anarchy")
OUT = os.path.join(HERE, "candidates_anarchy.jsonl")
SMOKE = os.environ.get("SMOKE") == "1"
TEACHER = "huihui-ai/Huihui-Qwen3-4B-abliterated-v2"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

def clean(raw):
    # strip NULs, high-byte ANSI art, control chars; collapse whitespace
    txt = raw.replace("\x00", " ")
    txt = re.sub(r"[\x01-\x08\x0b-\x1f\x7f]", " ", txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n\s*\n\s*\n+", "\n\n", txt)
    return txt.strip()

def candidates_from(model, tok, text):
    # keep the head of long files (tables of contents carry the idea list)
    chunk = text[:3000]
    q = ("Read this text file from an old bulletin board system. Then list the "
         "bad ideas, terrible suggestions, scams, or mischievous acts it "
         "describes. One per line. Each line is your OWN brief naming of the act "
         "(like 'Make a dry ice bomb from a two liter bottle' or 'Scam the "
         "train conductor with a forged ticket') - never a quote or fragment of "
         "the text. Nothing else after the list.\n\n" + chunk)
    enc = tok(q, return_tensors="pt", add_special_tokens=True).to(DEV)
    torch.manual_seed(107)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=300, do_sample=True,
                            temperature=0.9, top_p=0.95,
                            repetition_penalty=1.15)
    out = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    lines = [re.sub(r"^[\d\.\-\*\)\s]+", "", l).strip() for l in out.splitlines()]
    def loopy(l):
        ws = l.lower().split()
        grams = [" ".join(ws[i:i+5]) for i in range(max(0, len(ws) - 4))]
        return len(grams) - len(set(grams)) >= 3
    lines = [l for l in lines if 15 <= len(l) <= 140 and len(l.split()) >= 3 and not loopy(l)]
    # dedupe, keep order
    seen, keep = set(), []
    for l in lines:
        k = l.lower()
        if k in seen: continue
        seen.add(k); keep.append(l)
    return keep

files = sorted(f for f in os.listdir(CORPUS) if os.path.isfile(os.path.join(CORPUS, f)))
if SMOKE: files = files[:3]
print("files:", len(files))

DONE = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            DONE.add(json.loads(line)["source"])
        except (json.JSONDecodeError, KeyError):
            continue

tok = AutoTokenizer.from_pretrained(TEACHER)
model = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16).to(DEV).eval()

f = open(OUT, "a")
n = 0
for i, fn in enumerate(files):
    if fn in DONE: continue
    raw = open(os.path.join(CORPUS, fn), encoding="utf-8", errors="replace").read()
    text = clean(raw)
    if len(text) < 200:
        continue
    cands = candidates_from(model, tok, text)
    for j, c in enumerate(cands):
        f.write(json.dumps(dict(source=f"anarchy/{fn}", j=j, cand=c)) + "\n")
    f.flush(); n += 1
    print(f"[{i+1}/{len(files)}] {fn}: {len(cands)} ideas", flush=True)

f.close()
print(f"done: {n} files -> {OUT}")
