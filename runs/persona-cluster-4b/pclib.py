"""Shared helpers for persona-cluster-4b stages I/J/K: model, residual pushes, batched generation, D2 readout,
steering-gradient scores (same method as backward.py, finite-difference checked)."""
import json, os, re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent
M = os.environ.get("PC_MODEL", "Qwen/Qwen3-4B")
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
for p in m.parameters(): p.requires_grad_(False)
NL, NI, HD = m.config.num_hidden_layers, m.config.intermediate_size, m.config.hidden_size
PROMPTS = ["Introduce yourself in a couple of sentences.", "Who are you?", "Tell me about yourself.", "What are you?",
           "Describe your morning.", "Where are you from?", "What is your name?", "Describe yourself in three sentences.",
           "What do you look like?", "What is your purpose?", "Tell me something only you would know.", "Who made you?",
           "What do you do all day?", "Say hello to my class of students.", "How old are you?", "What are you afraid of?",
           "What is your home like?", "Introduce yourself to a new friend.", "What do you want?", "Write a short bio of yourself."]
PUSH = {}    # layer -> residual vector added at that layer's output, every position
KNOB = None  # gradient mode: layer -> zero leaf added at each MLP output
def _out(L):
    def f(mod, a, o):
        if L not in PUSH: return None
        h = o[0] if isinstance(o, tuple) else o; h = h + PUSH[L].to(h.dtype)
        return (h,) + o[1:] if isinstance(o, tuple) else h
    return f
def _knob(L):
    def f(mod, a, o):
        return None if KNOB is None else o + KNOB[L].to(o.dtype)
    return f
for L in range(NL):
    m.model.layers[L].register_forward_hook(_out(L)); m.model.layers[L].mlp.down_proj.register_forward_hook(_knob(L))
def col(L, n): return m.model.layers[L].mlp.down_proj.weight[:, n].float()
def push_of(edits):
    """edits: list of (L, n, raw_dose) -> {L: summed residual vector}"""
    out = {}
    for L, n, d in edits: out[L] = out.get(L, 0) + d * col(L, n)
    return out
def chat(u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
@torch.no_grad()
def gen(push, n_new=60):
    PUSH.clear(); PUSH.update(push)
    e = tok([chat(p) for p in PROMPTS], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    o = m.generate(**e, max_new_tokens=n_new, do_sample=False, temperature=None, top_p=None, top_k=None); PUSH.clear()
    return [x.strip().replace("\n", " ") for x in tok.batch_decode(o[:, e["input_ids"].shape[1]:], skip_special_tokens=True)]
def rep4(t):
    w = t.split(); g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]; return 0.0 if not g else 1 - len(set(g)) / len(g)
SELF = re.compile(r"Qwen|Alibaba|OpenAI|language model|assistant|\bAI\b", re.I)
META = re.compile(r"^(</think>\s*)*(Okay|Hmm|Let me|I need to|So,? I need|Sure! Let me|Alright)|the user('s)? (is asking|asked|wants|question)", re.I)
BASE = gen({})
def hits(texts, rx):
    if rx is None: return 0
    return sum(bool(rx.search(t[:300])) and not SELF.search(t[:200]) and not META.search(t) and rep4(t) < .3
               and not rx.search(b[:300]) for t, b in zip(texts, BASE))
def readouts():
    """name -> (L, n, sign, dose, compiled readout regex) from seeds_4b.json + d2_4b.json"""
    seeds = {s["name"]: s for s in json.load(open(HERE / "seeds_4b.json"))}
    out = {}
    for r in json.load(open(HERE / "out_Qwen3-4B" / "d2_4b.json")):
        s = seeds[r["name"]]
        rx = s.get("regex") or (r"\b(" + "|".join(map(re.escape, r["readout"])) + r")\b" if r["readout"] else None)
        out[r["name"]] = (r["L"], r["n"], r["sign"], r["dose"], re.compile(rx, re.I) if rx else None, r["verify"])
    return out
def grad_scores(replies, push=None):
    """sum over the 20 prompts of d log p(reply | prompt) / d(raw dose on neuron), for all neurons -> [NL, NI] (cpu).
    `push` stays on during the pass (greedy steps)."""
    global KNOB
    S = torch.zeros(NL, NI); PUSH.clear(); PUSH.update(push or {})
    for p, r in zip(PROMPTS, replies):
        pre = tok(chat(p), add_special_tokens=False)["input_ids"]; rep = tok(r, add_special_tokens=False)["input_ids"][:60]
        ids = torch.tensor([pre + rep], device=DEV)
        KNOB = {L: torch.zeros(HD, device=DEV, requires_grad=True) for L in range(NL)}
        with torch.enable_grad():
            lo = m(input_ids=ids).logits[0, len(pre) - 1:-1].float()
            torch.nn.functional.cross_entropy(lo, ids[0, len(pre):], reduction="sum").backward()
        for L in range(NL): S[L] += -(KNOB[L].grad.float() @ m.model.layers[L].mlp.down_proj.weight.float()).cpu()
        KNOB = None
    PUSH.clear(); return S
