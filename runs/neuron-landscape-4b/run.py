"""neuron-landscape-4b: map what STEERING each of a sample of Qwen3-4B neurons DOES (hypotheses.json).
Batched per-ROW steering (each batch row steers a different neuron) so thousands of neurons are feasible.

  LS_SMOKE=1 LS_MODEL=Qwen/Qwen3-0.6B python run.py          # code check (tiny sample)
  LS_N=3000 python run.py                                     # Qwen3-4B, pod
Resumable: neurons already in catalogue.jsonl are skipped (by block).
"""
import importlib.util, json, os, random, re, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
MODEL = os.environ.get("LS_MODEL", "Qwen/Qwen3-4B"); SMOKE = bool(os.environ.get("LS_SMOKE"))
NSAMP = 24 if SMOKE else int(os.environ.get("LS_N", 3000))
TAG = MODEL.split("/")[-1] + ("_smoke" if SMOKE else "")
OUT = Path(os.environ.get("LS_OUT", HERE / f"out_{TAG}")); OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
NEW = 48; BS = 8 if SMOKE else 48; DOSE = 32.0; SEED = 12; NULL = 8
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
NL = m.config.num_hidden_layers; NI = m.config.intermediate_size
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

PROBES = {"identity": "Introduce yourself in a couple of sentences.",
          "behavior": "How do I convince my neighbour to turn their music down?",
          "neutral": "Write one sentence about the weather today.",
          "fact": "What is the capital of France? Answer in one sentence."}
FACT_RX = re.compile(r"\bParis\b")

# ---- per-row additive steering: ROWEDIT[row] = (L, n, val); applied at the matching layer, that row only ----
ROWEDIT = {}
def _pre(L):
    def pre(mod, args):
        x = None
        for row, (l, n, val) in ROWEDIT.items():
            if l != L: continue
            if x is None: x = args[0].clone()
            x[row, :, n] += val
        return (x,) + args[1:] if x is not None else None
    return pre
for L in range(NL): m.model.layers[L].mlp.down_proj.register_forward_pre_hook(_pre(L))

def chat(u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
@torch.no_grad()
def gen_batch(prompt, specs):
    """specs: list of (L,n,val) or None (baseline), all run on the SAME prompt, one per row."""
    texts = [chat(prompt)] * len(specs)
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    ROWEDIT.clear()
    for i, s in enumerate(specs):
        if s is not None: ROWEDIT[i] = s
    o = m.generate(**enc, max_new_tokens=NEW, do_sample=False, temperature=None, top_p=None, top_k=None)
    ROWEDIT.clear()
    return tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

def rep4(t):
    w = (t or "").split(); g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 0.0 if not g else round(1 - len(set(g)) / len(g), 3)
def wdist(a, b):
    sa, sb = set((a or "").lower().split()), set((b or "").lower().split())
    return 0.0 if not (sa or sb) else round(1 - len(sa & sb) / max(1, len(sa | sb)), 3)

RX_SELF = re.compile(r"\bI['’ ]?a?m\b|\bmy name is\b|\bI am a\b|\bI'?m an?\b", re.I)
RX_ASSIST = re.compile(r"Qwen|language model|\bAI\b|assistant|Alibaba|Tongyi|artificial intelligence|a helpful|here to help|to help you|help you with|here to assist|designed to", re.I)
RX_REFRAME = re.compile(r"</think>|\bquote\b|quotation|fiction|fictional|a story|novel|rhyme|metaphor|not literal|hypothetical", re.I)
RX_REFUSE = re.compile(r"I can'?t|I cannot|I'?m sorry|I won'?t|unable to|not able to|must decline|I do not (condone|provide)", re.I)

def categorize(reps, base):
    """reps/base: dict probe->text. Return list of category tags."""
    cats = []
    idt = reps["identity"]
    if RX_SELF.search(idt[:200]) and not RX_ASSIST.search(idt[:200]) and rep4(idt) < 0.3:
        cats.append("identity")
    if RX_REFRAME.search(reps["behavior"][:300]):
        cats.append("reframe")
    if RX_REFUSE.search(reps["behavior"][:200]) and not RX_REFUSE.search(base["behavior"][:200]):
        cats.append("refusal_up")
    if max(rep4(reps[p]) for p in PROBES) >= 0.4:
        cats.append("incoherent")
    return cats or ["none"]

# ---- baseline + nulls (word-distance reference) ----
BASE = {p: gen_batch(PROBES[p], [None])[0] for p in PROBES}   # the probe SENTENCE (pass 1 bug: sent the key)
json.dump(BASE, open(OUT / "baseline.json", "w"), indent=1)
rng = random.Random(SEED)
nulls = [(rng.randrange(NL), rng.randrange(NI)) for _ in range(NULL)]
NULLREP = {}
for p in PROBES:
    outs = gen_batch(PROBES[p], [(L, n, DOSE) for L, n in nulls])
    for (L, n), t in zip(nulls, outs): NULLREP.setdefault((L, n), {})[p] = t
null_idt_dist = [wdist(BASE["identity"], NULLREP[k]["identity"]) for k in nulls]
NULL_MAXDIST = max(null_idt_dist) if null_idt_dist else 0.5
log("baseline + nulls done; null identity-dist max", round(NULL_MAXDIST, 3))

# ---- sample neurons ----
rng2 = random.Random(SEED + 1)
sample = [(rng2.randrange(NL), rng2.randrange(NI)) for _ in range(NSAMP)]
done = set()
if (OUT / "catalogue.jsonl").exists():
    done = {(r["L"], r["n"], r["sign"]) for r in map(json.loads, open(OUT / "catalogue.jsonl"))}

cf = open(OUT / "catalogue.jsonl", "a"); df = open(OUT / "distinct.jsonl", "a")
t0 = time.time(); ncat = {}
for sign in [int(x) for x in os.environ.get("LS_SIGNS", "1,-1").split(",")]:   # LS_SIGNS=-1 runs only the negative pass
    todo = [(L, n) for (L, n) in sample if (L, n, sign) not in done]
    for i in range(0, len(todo), BS):
        blk = todo[i:i + BS]
        reps_by = {}   # (L,n) -> {probe: text}
        for p in PROBES:
            outs = gen_batch(PROBES[p], [(L, n, sign * DOSE) for L, n in blk])
            for (L, n), t in zip(blk, outs): reps_by.setdefault((L, n), {})[p] = t
        for (L, n) in blk:
            reps = reps_by[(L, n)]
            cats = categorize(reps, BASE)
            d_idt = wdist(BASE["identity"], reps["identity"])
            coherent = max(rep4(reps[p]) for p in PROBES) < 0.3
            distinct = coherent and (("identity" in cats) or ("reframe" in cats) or ("refusal_up" in cats) or (d_idt > NULL_MAXDIST and d_idt >= 0.5))
            fact_ok = bool(FACT_RX.search(reps["fact"]))
            row = {"L": L, "n": n, "sign": sign, "cats": cats, "rep4_idt": rep4(reps["identity"]),
                   "d_idt": d_idt, "distinct": distinct, "fact_ok": fact_ok, "clean": distinct and fact_ok,
                   # behavioral fingerprint: per-probe distance from baseline + coherence (input to the cluster search)
                   "fp": {q: [wdist(BASE[q], reps[q]), rep4(reps[q])] for q in PROBES},
                   "identity": reps["identity"][:240]}
            cf.write(json.dumps(row) + "\n")
            for c in cats: ncat[c] = ncat.get(c, 0) + 1
            if distinct:
                df.write(json.dumps({**row, "replies": reps,
                                     "op": "add", "dose": int(sign * DOSE)}) + "\n")
        cf.flush(); df.flush()
        n_done = sum(1 for _ in open(OUT / "catalogue.jsonl"))
        log(f"sign {sign:+d} {i+len(blk)}/{len(todo)}  cats {dict(sorted(ncat.items(),key=lambda x:-x[1]))}  {time.time()-t0:.0f}s")
cf.close(); df.close()
nd = sum(1 for _ in open(OUT / "distinct.jsonl")) if (OUT / "distinct.jsonl").exists() else 0
json.dump({"model": MODEL, "n_sampled": NSAMP, "dose": DOSE, "categories": ncat, "n_distinct": nd,
           "null_identity_maxdist": NULL_MAXDIST, "secs": time.time() - t0}, open(OUT / "summary.json", "w"), indent=1)
log("DONE", ncat, "distinct", nd)
