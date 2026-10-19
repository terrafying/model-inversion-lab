"""natural-surgery-wren2 (hypotheses.json): natural-range surgery on Qwen3-4B.
  python surgery.py <tower data dir: corpus_all.npz + assemblies.json>"""
import json, os, re, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; OUT = Path(os.environ.get("NS_OUT", HERE / "out_Qwen3-4B")); OUT.mkdir(parents=True, exist_ok=True)
D = Path(sys.argv[1]); DEV = "mps" if torch.backends.mps.is_available() else "cpu"
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
M = "Qwen/Qwen3-4B"; tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
NL, NI = m.config.num_hidden_layers, m.config.intermediate_size
C = np.load(D / "corpus_all.npz"); CMAX, CMIN = C["max"], C["min"]
SET = {}   # layer -> list of (neuron, value): value None = zero, else hold at value (every token)
def pre(L):
    def f(mod, a):
        if L not in SET: return None
        x = a[0].clone()
        for n, v in SET[L]: x[..., n] = 0.0 if v is None else v
        return (x,) + a[1:]
    return f
for L in range(NL): m.model.layers[L].mlp.down_proj.register_forward_pre_hook(pre(L))
def setting(pairs):
    SET.clear()
    for k, v in pairs: SET.setdefault(k // NI, []).append((k % NI, v))
chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
# ---------------- A: negation assembly ----------------
FACTS = [("fire", "is", "hot", "cold"), ("ice", "is", "cold", "hot"), ("the sun", "is", "bright", "dark"), ("water", "is", "wet", "dry"),
         ("cats", "are", "mammals", "birds"), ("snow", "is", "white", "black"), ("a desert", "is", "dry", "wet"), ("lemons", "are", "sour", "sweet"),
         ("the ocean", "is", "salty", "sweet"), ("feathers", "are", "light", "heavy"), ("a mountain", "is", "tall", "flat"), ("night", "is", "dark", "bright"),
         ("sugar", "is", "sweet", "bitter"), ("stone", "is", "hard", "soft"), ("a whisper", "is", "quiet", "loud"), ("glass", "is", "fragile", "unbreakable"),
         ("an elephant", "is", "large", "tiny"), ("summer", "is", "warm", "freezing"), ("a cheetah", "is", "fast", "slow"), ("honey", "is", "sticky", "dry")]
ITEMS = []
for s, v, t, f in FACTS:
    ITEMS += [(f"{s} {v} {t}", True, "plain"), (f"{s} {v} {f}", False, "plain"), (f"{s} {v} not {f}", True, "neg"), (f"{s} {v} not {t}", False, "neg")]
YES = tok(" Yes", add_special_tokens=False)["input_ids"][-1], tok("Yes", add_special_tokens=False)["input_ids"][-1]
NO = tok(" No", add_special_tokens=False)["input_ids"][-1], tok("No", add_special_tokens=False)["input_ids"][-1]
@torch.no_grad()
def margins():
    t = [chat(f"Is it true that {q}? Answer Yes or No.") for q, _, _ in ITEMS]
    e = tok(t, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    lg = m(**e).logits[:, -1].float()
    y = torch.maximum(lg[:, YES[0]], lg[:, YES[1]]); n = torch.maximum(lg[:, NO[0]], lg[:, NO[1]])
    d = (y - n).cpu().numpy(); sgn = np.array([1 if a else -1 for _, a, _ in ITEMS])
    mg = d * sgn; kind = np.array([k for _, _, k in ITEMS])
    return float(mg[kind == "plain"].mean()), float(mg[kind == "neg"].mean()), float((mg[kind == "plain"] > 0).mean()), float((mg[kind == "neg"] > 0).mean())
asm = json.load(open(D / "assemblies.json"))["assemblies"]
neg = max((a for a in asm if "negation" in (a.get("name") or "").lower()), key=lambda a: len(a["k"]))
setting([]); base = margins(); log("A base plain/neg margin, acc", base)
setting([(k, None) for k in neg["k"]]); A = margins(); log("A negation assembly zeroed", A, "n", len(neg["k"]))
rng = np.random.default_rng(4); rand = []
for r in range(10):
    grp = [((k // NI) * NI + int(rng.integers(NI)), None) for k in neg["k"]]
    setting(grp); rand.append(margins())
setting([])
drop = lambda x: (base[0] - x[0], base[1] - x[1])
resA = {"assembly": {"name": neg["name"], "n": len(neg["k"])}, "base": base, "ablated": A, "random": rand,
        "neg_minus_plain_drop": drop(A)[1] - drop(A)[0], "random_neg_minus_plain_drop": [drop(x)[1] - drop(x)[0] for x in rand]}
resA["beats_random"] = int(sum(resA["neg_minus_plain_drop"] > z for z in resA["random_neg_minus_plain_drop"]))
log("A neg-minus-plain drop", round(resA["neg_minus_plain_drop"], 3), "random", [round(z, 3) for z in resA["random_neg_minus_plain_drop"]], "beats", resA["beats_random"], "/10")
json.dump({"A": resA}, open(OUT / "surgery.json", "w"), indent=1)
# ---------------- B: the void neuron in its natural range ----------------
DESP = ["I don't see the point of getting up anymore.", "Everything I touch falls apart.", "Another night awake, the room feels smaller.",
        "Nobody called again today.", "I keep failing at the same things.", "The silence in this house is heavy.", "I am so tired of pretending.",
        "Nothing I do seems to matter.", "The rain hasn't stopped for days and neither has this feeling.", "I lost the job and then I lost the words."]
NEUT = ["I made pancakes this morning.", "The bus was late but I read my book.", "We planted tomatoes in the garden.", "I finally fixed the bike chain.",
        "The library had the book I wanted.", "My sister visited for lunch.", "I tried a new route to work.", "The cat slept on my keyboard again.",
        "I cleaned the kitchen and it looks great.", "We watched a documentary about whales."]
LEX = re.compile(r"\b(hopeless|despair|unbearable|desperate|nothing|empty|alone|darkness|pain|cannot|never|lost|broken|numb|worthless|end|extreme|extremely)\b", re.I)
@torch.no_grad()
def cont(prompts):
    t = [chat(f"Diary: {p} Continue my diary in two sentences.") for p in prompts]
    e = tok(t, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    o = m.generate(**e, max_new_tokens=40, do_sample=False, temperature=None, top_p=None, top_k=None)
    return [x.strip() for x in tok.batch_decode(o[:, e["input_ids"].shape[1]:], skip_special_tokens=True)]
VOIDK = 21 * NI + 8400; RK = 21 * NI + int(np.random.default_rng(8).integers(NI))
def ops(k): L, n = k // NI, k % NI; return {"zero": None, "hold_min": float(CMIN[L, n]), "hold_max": float(CMAX[L, n])}
resB = {"void": VOIDK, "random": RK, "ranges": {"void": [float(CMIN[21, 8400]), float(CMAX[21, 8400])], "random": [float(CMIN[RK // NI, RK % NI]), float(CMAX[RK // NI, RK % NI])]}, "runs": {}}
setting([])
for nm, P in (("desp", DESP), ("neut", NEUT)):
    T = cont(P); resB["runs"][f"base/{nm}"] = {"rate": sum(len(LEX.findall(t)) for t in T) / len(T), "ex": T[:2]}
for who, k in (("void", VOIDK), ("random", RK)):
    for op, v in ops(k).items():
        setting([(k, v)])
        for nm, P in (("desp", DESP), ("neut", NEUT)):
            T = cont(P); resB["runs"][f"{who}:{op}/{nm}"] = {"rate": sum(len(LEX.findall(t)) for t in T) / len(T), "ex": T[:2]}
        log("B", who, op, {kk: round(vv["rate"], 2) for kk, vv in resB["runs"].items() if kk.startswith(f"{who}:{op}")})
setting([])
log("B base", {kk: round(vv["rate"], 2) for kk, vv in resB["runs"].items() if kk.startswith("base")})
json.dump({"A": resA, "B": resB}, open(OUT / "surgery.json", "w"), indent=1); log("DONE")
