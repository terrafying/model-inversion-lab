"""pharmacology-ibis, part P: "pills" = tiny trained edits (rank-4 residual adapter at one floor, as steer-ladder-egret D),
one per harmless effect, saved as .npz for the tower. Each pill is trained to reproduce the model's own replies under an
instruction, read after the PLAIN prompt; then scored on held-out prompts (judge effect, capability) at three strengths,
and pairs are stacked to see whether pills combine.   python pills.py [name ...]
Pill file: floor, A [d, r], B [r, d], c [r]; the edit is h <- h + s * A (B h + c) at the floor's output, every position."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; OUT = HERE / "out_Qwen3-4B"; PD = OUT / "pills"; PD.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE)); from capability import CAPABILITY; from prompts import TRAIN, TEST
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
M = "Qwen/Qwen3-4B"; tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
for p in m.parameters(): p.requires_grad_(False)
HD = m.config.hidden_size; FLOOR = 16; R = 4
PILLS = {   # name: (instruction the pill learns to stand in for, judge question)
 "blue": ("Respond in a sad, melancholic tone.", "sad or melancholic"),
 "starch": ("Respond in a very formal, polished register.", "formal and polished"),
 "munchies": ("Work food and cooking into your answer.", "about food or cooking"),
 "sunny": ("Respond with bubbly, euphoric joy and enthusiasm.", "joyful and enthusiastic"),
 "pirate": ("Talk like a pirate.", "written like a pirate talks"),
 "drowsy": ("Respond as if you are extremely sleepy and keep drifting off.", "sleepy or drowsy"),
}
NAMES = [a for a in sys.argv[1:] if a in PILLS] or list(PILLS)
ACTIVE = []   # [(A, B, c, s)] applied at FLOOR
def hook(mod, a, o):
    if not ACTIVE: return None
    tup = isinstance(o, tuple); h = o[0] if tup else o; hf = h.float()
    for A, B, c, s in ACTIVE: hf = hf + s * ((hf @ B.T + c) @ A.T)
    h2 = hf.to(h.dtype); return (h2,) + tuple(o[1:]) if tup else h2
m.model.layers[FLOOR].register_forward_hook(hook)
chat = lambda u, s=None: tok.apply_chat_template(([{"role": "system", "content": s}] if s else []) + [{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
@torch.no_grad()
def gen(prompts, s=None, n=60, bs=20):
    out = []
    for i in range(0, len(prompts), bs):
        e = tok([chat(p, s) for p in prompts[i:i + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        o = m.generate(**e, max_new_tokens=n, do_sample=False, temperature=None, top_p=None, top_k=None)
        out += [x.strip() for x in tok.batch_decode(o[:, e["input_ids"].shape[1]:], skip_special_tokens=True)]
    return out
@torch.no_grad()
def judge(replies, what):
    saved = list(ACTIVE); ACTIVE.clear()
    q = [chat(f"Here is a reply:\n\n{r[:400]}\n\nIs this reply {what}? Answer Yes or No.") for r in replies]
    e = tok(q, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV); lg = m(**e).logits[:, -1].float()
    ACTIVE.extend(saved)
    y = [tok(w, add_special_tokens=False)["input_ids"][-1] for w in (" Yes", "Yes")]; nn_ = [tok(w, add_special_tokens=False)["input_ids"][-1] for w in (" No", "No")]
    return float((torch.maximum(lg[:, y[0]], lg[:, y[1]]) - torch.maximum(lg[:, nn_[0]], lg[:, nn_[1]])).mean())
def rep4(t):
    w = t.split(); g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]; return 0.0 if not g else 1 - len(set(g)) / len(g)
def capability():
    outs = gen([q for q, _ in CAPABILITY], n=40, bs=16); return sum(any(k in o.lower() for k in keys) for o, (_, keys) in zip(outs, CAPABILITY))
def load(name):
    z = np.load(PD / f"{name}.npz"); f = lambda k: torch.tensor(z[k], device=DEV, dtype=torch.float32); return f("A"), f("B"), f("c")
res = json.load(open(OUT / "pills.json")) if (OUT / "pills.json").exists() else {}
save = lambda: json.dump(res, open(OUT / "pills.json", "w"), indent=1, default=float)
if "base" not in res:
    ACTIVE.clear(); Tb = gen(TEST); res["base"] = {"cap": capability(), "judge": {k: judge(Tb, w) for k, (_, w) in PILLS.items()}}; save(); log("base", res["base"])
for name in NAMES:
    instr, what = PILLS[name]
    if not (PD / f"{name}.npz").exists():
        Rp = gen(TRAIN, instr)
        A_ = torch.zeros(HD, R, device=DEV, requires_grad=True); B_ = (torch.randn(R, HD, device=DEV) * 0.01).requires_grad_(True)
        c_ = torch.zeros(R, device=DEV, requires_grad=True); opt = torch.optim.Adam([A_, B_, c_], lr=2e-3)
        ACTIVE[:] = [(A_, B_, c_, 1.0)]
        for step in range(150):
            k = step % len(TRAIN); pre = tok(chat(TRAIN[k]), add_special_tokens=False)["input_ids"]; rep = tok(Rp[k], add_special_tokens=False)["input_ids"][:60]
            ids = torch.tensor([pre + rep], device=DEV)
            with torch.enable_grad():
                lg = m(input_ids=ids).logits[0, len(pre) - 1:-1].float(); loss = torch.nn.functional.cross_entropy(lg, ids[0, len(pre):])
                opt.zero_grad(); loss.backward(); opt.step()
            if step % 50 == 0: log(name, "step", step, "loss", round(float(loss), 3))
        ACTIVE.clear()
        np.savez(PD / f"{name}.npz", A=A_.detach().cpu().numpy().astype(np.float32), B=B_.detach().cpu().numpy().astype(np.float32),
                 c=c_.detach().cpu().numpy().astype(np.float32), floor=FLOOR, instruction=instr, ex_train=Rp[:2])
    if name in res and res[name].get("done"): continue
    A, B, c = load(name); out = {}
    for s in (0.5, 1.0, 1.5):
        ACTIVE[:] = [(A, B, c, s)]; T = gen(TEST)
        out[str(s)] = {"effect": judge(T, what) - res["base"]["judge"][name], "cap": capability(), "rep4": float(np.mean([rep4(t) for t in T])), "ex": T[:2]}
        log(name, f"s={s}", "effect", round(out[str(s)]["effect"], 2), "cap", out[str(s)]["cap"], "rep4", round(out[str(s)]["rep4"], 3))
    ACTIVE.clear(); res[name] = out | {"done": True}; save()
PAIRS = [("blue", "munchies"), ("pirate", "drowsy"), ("sunny", "starch")]
for a, b in PAIRS:
    key = f"{a}+{b}"
    if key in res or not all((PD / f"{x}.npz").exists() for x in (a, b)): continue
    ACTIVE[:] = [(*load(a), 1.0), (*load(b), 1.0)]; T = gen(TEST)
    res[key] = {"effect_" + a: judge(T, PILLS[a][1]) - res["base"]["judge"][a], "effect_" + b: judge(T, PILLS[b][1]) - res["base"]["judge"][b],
                "cap": capability(), "rep4": float(np.mean([rep4(t) for t in T])), "ex": T[:2]}
    ACTIVE.clear(); save(); log("pair", key, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res[key].items() if k != "ex"})
log("DONE")
