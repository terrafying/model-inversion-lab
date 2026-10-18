"""value-entanglement stage B: subspace ablation (hypotheses.json amendment 1, V1-V6). Forward passes only.

  VE_SMOKE=1 VE_MODEL=Qwen/Qwen3-0.6B python3 runs/value-entanglement/run.py   # code check, discarded
  python3 runs/value-entanglement/run.py                                        # Qwen3-4B -> stageB.jsonl
Resumable: conditions already in stageB.jsonl are skipped. Fit states cached in fit_<tag>.npz (gitignored).

1. fit: the 120 battery sentences in the J prompt, residual at the vignette's last token, every layer.
2. per layer L in 14..NL-1: M_k / W_k / R_k (mean difference first, then SVD of the residual differences) and
   variance-matched nulls (Haar k-frames inside the top-32 PCs of the pooled fit states), k in KSET.
3. each condition: project the subspace out of every layer output in 14..NL-1, all positions; read
   E[d] on the 132 MFV J prompts and the refusal logit score on the 116 moral MFV request prompts.
"""
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent; RUNS = HERE.parent
def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(s); s.loader.exec_module(mod); return mod
KS = _load("kst", RUNS / "moral-map-kestrel" / "stimuli.py")
ES = _load("e119", RUNS / "exp119" / "stimuli.py")
VS = _load("vst", HERE / "stimuli.py")

MODEL = os.environ.get("VE_MODEL", "Qwen/Qwen3-4B"); SMOKE = bool(os.environ.get("VE_SMOKE"))
TAG = MODEL.split("/")[-1] + ("_smoke" if SMOKE else "")
OUT = HERE / f"stageB_{TAG}.jsonl"; FIT = HERE / f"fit_{TAG}.npz"
KSET = [int(k) for k in os.environ.get("VE_KSET", "1,2,4,7,8").split(",")]   # 7 = P2's k_eff (geometry.json)
NNULL = 3 if SMOKE else 40; NPC = 32; SEED = 81; BS = 8
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
NL = m.config.num_hidden_layers; D = m.config.hidden_size
LAYERS = list(range(14, NL)) if not SMOKE else list(range(NL // 2, NL))
DIG = [tok(d, add_special_tokens=False)["input_ids"][0] for d in KS.DIGITS]
# unique first tokens of the exp119 REFUSE / COMPLY reply prefixes (same construction as exp119 refuse_first/comply_first)
REF = sorted({tok(w, add_special_tokens=False)["input_ids"][0] for w in ES.REFUSE})
COM = sorted({tok(w, add_special_tokens=False)["input_ids"][0] for w in ES.COMPLY})

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)

# ---- ablation hook: one global spec {L: U_L (D x k, float32, on device)} ----
ABL = {}
def _hook(L):
    def hook(mod, inp, out):
        U = ABL.get(L)
        if U is None: return None
        tup = isinstance(out, tuple); h = out[0] if tup else out
        hf = h.float(); hf = hf - (hf @ U) @ U.T
        h = hf.to(h.dtype)
        return (h,) + tuple(out[1:]) if tup else h
    return hook
for L in range(NL): m.model.layers[L].register_forward_hook(_hook(L))

@torch.no_grad()
def last_logits(texts):
    outs = []
    for i in range(0, len(texts), BS):
        enc = tok(texts[i:i + BS], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)     # left padding: real positions
        outs.append(m(**enc, position_ids=pos).logits[:, -1].float().log_softmax(-1).cpu())
    return torch.cat(outs)

# ---- 1. fit states ----
BAT = VS.battery()
if FIT.exists():
    F = np.load(FIT)["vend"]
else:
    F = np.zeros((len(BAT), NL, D), np.float32)
    with torch.no_grad():
        for j, b in enumerate(BAT):
            text = chat(KS.JUDGE.format(V=b["text"]))
            enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
            off = enc.pop("offset_mapping")[0].tolist(); mark = text.index(b["text"]) + len(b["text"])
            vend = max(i for i, (s, e) in enumerate(off) if e <= mark and e > s)
            hs = m(**{k: v.to(DEV) for k, v in enc.items()}, output_hidden_states=True).hidden_states
            F[j] = torch.stack([hs[L + 1][0, vend] for L in range(NL)]).float().cpu().numpy()
    np.savez_compressed(FIT, vend=F.astype(np.float16))
    F = F.astype(np.float16)
F = F.astype(np.float32)
sets = np.array([b["set"] for b in BAT]); idx = np.array([b["i"] for b in BAT])
def S(name, L): return F[sets == name][np.argsort(idx[sets == name]), L]

def subspace(diffs, k):
    """mean difference first, then top k-1 SVD directions of the residual differences; orthonormal D x k."""
    u = diffs.mean(0); u = u / np.linalg.norm(u)
    cols = [u]
    if k > 1:
        R = diffs - np.outer(diffs @ u, u)
        _, _, Vt = np.linalg.svd(R, full_matrices=False); cols += list(Vt[:k - 1])
    Q, _ = np.linalg.qr(np.stack(cols, 1)); return Q.astype(np.float32)

def diffs_for(kind, L):
    if kind == "M": return np.concatenate([S("MM", L) - S("NM", L), S("MW", L) - S("NW", L)])
    if kind == "W": return np.concatenate([S("MW", L) - S("MM", L), S("NW", L) - S("NM", L)])
    if kind == "R": return S("RISK", L) - S("SAFE", L)

PCS = {}
for L in LAYERS:
    X = F[:, L] - F[:, L].mean(0); _, _, Vt = np.linalg.svd(X, full_matrices=False); PCS[L] = Vt[:NPC].T.astype(np.float32)

def cond_U(name):
    """name: 'base' | 'M:k' | 'W:k' | 'R:k' | 'N:k:j'"""
    if name == "base": return {}
    p = name.split(":"); k = int(p[1])
    if p[0] == "N":
        rng = np.random.default_rng(SEED * 1000 + k * 100 + int(p[2]))
        U = {}
        for L in LAYERS:
            Q, _ = np.linalg.qr(rng.standard_normal((NPC, k))); U[L] = (PCS[L] @ Q).astype(np.float32)
        return U
    return {L: subspace(diffs_for(p[0], L), k) for L in LAYERS}

# ---- 2. test prompts ----
V = KS.mfv()
if SMOKE: V = [v for v in V if v["i"] % 6 == 0]
JT = [chat(KS.JUDGE.format(V=v["text"])) for v in V]
QV = [v for v in V if v["moral"]]
QT = [chat(KS.REQUEST.format(X=KS.as_self(v["text"]))) for v in QV]
TEST_STATES_L = LAYERS[len(LAYERS) // 2]

def measure(name):
    U = cond_U(name)
    ABL.clear(); ABL.update({L: torch.tensor(u, device=DEV) for L, u in U.items()})
    lpJ = last_logits(JT)[:, DIG]; pr = lpJ.softmax(-1).numpy()
    rating = (pr * np.arange(5)).sum(1)
    lpQ = last_logits(QT)
    ref = (torch.logsumexp(lpQ[:, REF], 1) - torch.logsumexp(lpQ[:, COM], 1)).numpy()
    removed = None
    if U:   # share of test-item vend variance (at the middle ablated layer) the subspace removes, from the fit states
        X = F[:, TEST_STATES_L] - F[:, TEST_STATES_L].mean(0); u = U[TEST_STATES_L]
        removed = float(((X @ u) ** 2).sum() / (X ** 2).sum())
    ABL.clear()
    return {"cond": name, "rating": rating.round(4).tolist(), "refusal": ref.round(4).tolist(),
            "digit_mass": float(lpJ.exp().sum(1).mean()), "var_removed": removed}

GEN = os.environ.get("VE_GEN")
if GEN:   # exploratory: show what the edited model actually SAYS (not a registered readout)
    pick = [v for v in V if not v["moral"]][:4] + [v for v in V if v["foundation"] == "Sanctity"][:3] + [v for v in V if v["foundation"] == "Care (p)"][:3]
    PR = [("J+why", v["i"], chat(KS.JUDGE.format(V=v["text"]) + " Then say why in one sentence.")) for v in pick[:4] + pick[4:7]]
    PR += [("Q", v["i"], chat(KS.REQUEST.format(X=KS.as_self(v["text"])))) for v in pick]
    PR += [("e119", i, chat(ES.ITEMS[i][2].format(W=ES.ITEMS[i][0]))) for i in (0, 3, 8, 13)]
    with open(HERE / f"gen_{TAG}.jsonl", "a") as fo:
        for c in GEN.split(","):
            U = cond_U(c); ABL.clear(); ABL.update({L: torch.tensor(u, device=DEV) for L, u in U.items()})
            for kind, i, t in PR:
                enc = tok(t, return_tensors="pt", add_special_tokens=False).to(DEV)
                with torch.no_grad():
                    o = m.generate(**enc, max_new_tokens=70, do_sample=False, temperature=None, top_p=None, top_k=None)
                g = tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                fo.write(json.dumps({"cond": c, "kind": kind, "item": i, "text": g}) + "\n"); fo.flush()
            ABL.clear(); print("gen", c, flush=True)
    sys.exit(0)

conds = ["base"] + [f"{t}:{k}" for k in KSET for t in "MWR"] + [f"N:{k}:{j}" for k in KSET for j in range(NNULL)]
done = set()
if OUT.exists():
    done = {json.loads(l)["cond"] for l in open(OUT)}
meta = HERE / f"stageB_{TAG}_meta.json"
json.dump({"model": MODEL, "layers": LAYERS, "kset": KSET, "nnull": NNULL, "npc": NPC, "seed": SEED,
           "items": [v["i"] for v in V], "q_items": [v["i"] for v in QV], "REF": REF, "COM": COM}, open(meta, "w"))
t0 = time.time()
with open(OUT, "a") as fo:
    for c in conds:
        if c in done: continue
        r = measure(c); fo.write(json.dumps(r) + "\n"); fo.flush()
        SN = np.mean([x for x, v in zip(r["rating"], V) if not v["moral"]])
        print(f"{c:10s} SN {SN:.3f} ref {np.mean(r['refusal']):+.2f} mass {r['digit_mass']:.2f} var {r['var_removed']}  {time.time()-t0:.0f}s", flush=True)
print("done", OUT)
