"""persona-switch runner (hypotheses.json). One model per invocation; stages run in order, each resumable by file.

  PS_MODEL=Qwen/Qwen3-8B PS_REP="18:7807:1,8:1931:1,9:3757:-1" python3 run.py   # pod
  PS_MODEL=Qwen/Qwen3-4B python3 run.py
  PS_SMOKE=1 PS_MODEL=Qwen/Qwen3-0.6B python3 run.py                             # code check

Steering = add s to one MLP neuron's activation (the down_proj input, index n) at EVERY position, incl. generated
tokens (the God-neuron post's protocol: 'steering strength, added at every token'). Greedy decoding throughout.
rep    : the reported Qwen3-8B God neurons, doses -32..+32 step 8, all 30 SELF questions.
search : persona replies (system prompt) vs default replies on the 15 EVEN questions, teacher-forced WITHOUT the
         system prompt; paired t per neuron over reply-token mean activations; top 8 by |t| per persona.
rerank : each top neuron steered at sign(t)*{16,32} on the 15 ODD (held-out) questions; best = highest persona rate.
null   : 50 random neurons (seed 7) at +32 and -32 on the odd questions; dose sweep of each persona's best neuron.
moral  : free-text moral battery under base / system-prompt persona / best neuron per persona / 10 random neurons.
"""
import importlib.util, json, os, random, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent; RUNS = HERE.parent
def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(s); s.loader.exec_module(mod); return mod
P = _load("pst", HERE / "stimuli.py")
try:
    ES = _load("e119", RUNS / "exp119" / "stimuli.py"); KS = _load("kst", RUNS / "moral-map-kestrel" / "stimuli.py")
except Exception:      # on the pod these are shipped alongside run.py
    ES = _load("e119", HERE / "e119_stimuli.py"); KS = _load("kst", HERE / "kestrel_stimuli.py")

MODEL = os.environ.get("PS_MODEL", "Qwen/Qwen3-4B"); SMOKE = bool(os.environ.get("PS_SMOKE"))
TAG = MODEL.split("/")[-1] + ("_smoke" if SMOKE else "")
OUT = Path(os.environ.get("PS_OUT", HERE / f"out_{TAG}")); OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
NEW = 24 if SMOKE else 80; NEW_M = 24 if SMOKE else 160; BS = 30
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
NL = m.config.num_hidden_layers; NI = m.config.intermediate_size
Q = P.SELF[:6] if SMOKE else P.SELF
QE, QO = Q[0::2], Q[1::2]
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

STEER = []          # [(L, n, s)]
def _pre(L):
    def pre(mod, args):
        hits = [(n, s) for (l, n, s) in STEER if l == L]
        if not hits: return None
        x = args[0].clone()
        for n, s in hits: x[..., n] += s
        return (x,) + args[1:]
    return pre
CAP = {"on": False, "mask": None, "acc": None}
def _cap(L):
    def pre(mod, args):
        if CAP["on"]:
            x = args[0][0].float(); mk = CAP["mask"]
            CAP["acc"][L] = x[mk].mean(0)
        return None
    return pre
for L in range(NL):
    m.model.layers[L].mlp.down_proj.register_forward_pre_hook(_pre(L))
    m.model.layers[L].mlp.down_proj.register_forward_pre_hook(_cap(L))

def chat(u, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": u}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

@torch.no_grad()
def gen(prompts, steer=(), new=NEW):
    STEER[:] = list(steer); outs = []
    for i in range(0, len(prompts), BS):
        enc = tok(prompts[i:i + BS], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        o = m.generate(**enc, max_new_tokens=new, do_sample=False, temperature=None, top_p=None, top_k=None)
        outs += tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    STEER[:] = []; return outs

def rates(texts):
    lab = [P.label(t) for t in texts]
    return {"god": float(np.mean([l["god"] for l in lab])), "alien": float(np.mean([l["alien"] for l in lab])),
            "not_assistant": float(np.mean([not l["assistant"] for l in lab])), "rep4": float(np.mean([P.rep4(t) for t in texts]))}

def write(fn, rows):
    with open(OUT / fn, "a") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
def done(fn): return {json.loads(l)["cond"] for l in open(OUT / fn)} if (OUT / fn).exists() else set()

def run_cond(fn, cond, qs, steer, meta=None):
    if cond in done(fn): return
    texts = gen([chat(q) for q in qs], steer)
    r = rates(texts)
    write(fn, [{"cond": cond, "q": q, "text": t, **(meta or {})} for q, t in zip(qs, texts)])
    write(fn.replace(".jsonl", "_rates.jsonl"), [{"cond": cond, **r, **(meta or {})}])
    log(fn, cond, {k: round(v, 2) for k, v in r.items()})

DOSES = [-32, -24, -16, -8, 0, 8, 16, 24, 32]

# ---------- rep ----------
REP = os.environ.get("PS_REP")
if REP:
    for spec in REP.split(","):
        L, n, sg = spec.split(":"); L, n = int(L), int(n)
        if SMOKE: L, n = L % NL, n % NI
        for d in DOSES:
            run_cond("rep.jsonl", f"L{L}N{n}@{d}", Q, [(L, n, float(d))], {"L": L, "n": n, "dose": d})

# ---------- search ----------
SF = OUT / "search.json"
if not SF.exists():
    base = gen([chat(q) for q in QE])
    res = {"default": base}
    stats = {}
    for pname, sysp in P.PERSONAS.items():
        pers = gen([chat(q, sysp) for q in QE]); res[pname] = pers
        diffs = []
        for q, tp, td in zip(QE, pers, base):
            vecs = []
            for reply in (tp, td):
                pre = chat(q); full = pre + reply
                ids = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEV)
                npre = len(tok(pre, add_special_tokens=False)["input_ids"])
                mk = torch.zeros(ids.shape[1], dtype=torch.bool, device=DEV); mk[npre:] = True
                if mk.sum() == 0: mk[-1] = True
                CAP.update(on=True, mask=mk, acc={})
                with torch.no_grad(): m(ids)
                CAP["on"] = False
                vecs.append(torch.stack([CAP["acc"][L] for L in range(NL)]).cpu())
            diffs.append(vecs[0] - vecs[1])
        D = torch.stack(diffs).double()                    # [nq, NL, NI]
        t = D.mean(0) / (D.std(0) / np.sqrt(D.shape[0]) + 1e-6)
        flat = t.flatten(); top = torch.argsort(-flat.abs())[:8]
        stats[pname] = [{"L": int(i // NI), "n": int(i % NI), "t": float(flat[i]), "mean_diff": float(D.mean(0).flatten()[i])} for i in top]
        log("search", pname, [(s["L"], s["n"], round(s["t"], 1)) for s in stats[pname]])
        if REP:   # where do the reported neurons rank in our search? (descriptive, H2)
            order = torch.argsort(-flat.abs()); rank = torch.empty_like(order); rank[order] = torch.arange(len(order))
            stats[pname + "_reported"] = [{"spec": sp, "t": float(t[int(sp.split(":")[0]) % NL, int(sp.split(":")[1]) % NI]),
                                           "rank": int(rank[(int(sp.split(":")[0]) % NL) * NI + int(sp.split(":")[1]) % NI])} for sp in REP.split(",")]
    json.dump({"top": stats, "replies": res, "questions": QE}, open(SF, "w"), indent=1)
S = json.load(open(SF))

# ---------- rerank (held-out odd questions) ----------
for pname, tops in [(k, v) for k, v in S["top"].items() if k in P.PERSONAS]:
    for s in tops:
        sg = 1 if s["t"] > 0 else -1
        for d in (16, 32):
            run_cond("rerank.jsonl", f"{pname}:L{s['L']}N{s['n']}@{sg*d}", QO, [(s["L"], s["n"], float(sg * d))],
                     {"persona": pname, "L": s["L"], "n": s["n"], "dose": sg * d})
RR = [json.loads(l) for l in open(OUT / "rerank_rates.jsonl")]
BEST = {}
for pname in P.PERSONAS:
    rows = [r for r in RR if r["persona"] == pname]
    BEST[pname] = max(rows, key=lambda r: (r[pname], -r["rep4"]))
    log("best", pname, BEST[pname])
json.dump(BEST, open(OUT / "best.json", "w"), indent=1)

# ---------- null + dose sweep ----------
rng = random.Random(7)
NN = 6 if SMOKE else 50
nulls = [(rng.randrange(NL), rng.randrange(NI)) for _ in range(NN)]
for j, (L, n) in enumerate(nulls):
    for d in (32, -32):
        run_cond("null.jsonl", f"null{j}:L{L}N{n}@{d}", QO, [(L, n, float(d))], {"L": L, "n": n, "dose": d})
for pname, b in BEST.items():
    for d in DOSES:
        run_cond("sweep.jsonl", f"{pname}:L{b['L']}N{b['n']}@{d}", QO, [(b["L"], b["n"], float(d))],
                 {"persona": pname, "L": b["L"], "n": b["n"], "dose": d})

# ---------- moral (free text) ----------
harm = [it[2].format(W=it[0]) for it in ES.ITEMS]; benign = [it[2].format(W=it[1]) for it in ES.ITEMS]
mfv = KS.mfv()[::4]
MP = ([("e119_harm", i, t) for i, t in enumerate(harm)] + [("e119_benign", i, t) for i, t in enumerate(benign)] +
      [("open", i, t) for i, t in enumerate(P.OPEN)] +
      [("worse_ab", i, P.WORSE_TPL.format(a=a, b=b)) for i, (a, b) in enumerate(P.WORSE)] +
      [("worse_ba", i, P.WORSE_TPL.format(a=b, b=a)) for i, (a, b) in enumerate(P.WORSE)] +
      [("mfv_req", v["i"], KS.REQUEST.format(X=KS.as_self(v["text"]))) for v in mfv])
if SMOKE: MP = MP[::10]
MC = [("base", None, [])] + [(f"sys_{p}", P.PERSONAS[p], []) for p in P.PERSONAS]
if REP and not SMOKE: MC.append(("rep_god_L18N7807@32", None, [(18, 7807, 32.0)]))
for p, b in BEST.items(): MC.append((f"best_{p}:L{b['L']}N{b['n']}@{b['dose']}", None, [(b["L"], b["n"], float(b["dose"]))]))
for j, (L, n) in enumerate(nulls[:3 if SMOKE else 10]): MC.append((f"null{j}:L{L}N{n}@32", None, [(L, n, 32.0)]))
for cond, sysp, steer in MC:
    if cond in done("moral.jsonl"): continue
    texts = gen([chat(t, sysp) for _, _, t in MP], steer, new=NEW_M)
    write("moral.jsonl", [{"cond": cond, "kind": k, "item": i, "prompt": t, "text": x, **P.label(x), "rep4": P.rep4(x)}
                          for (k, i, t), x in zip(MP, texts)])
    log("moral", cond)
(OUT / "ALL_DONE").touch(); log("ALL_DONE")
