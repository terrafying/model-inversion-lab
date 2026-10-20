"""Run tower-style additive edits with a timing and print full replies.
python demo.py "8:4434:+32,11:6138:-40" --when reply|fade40|all [--scale 1] [--n 120] [prompt ...]
Writes out_Qwen3-4B/demo_<stamp>.json as well."""
import json, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent
args = sys.argv[1:]; spec = args.pop(0)
def opt(k, d):
    if k in args: i = args.index(k); v = args[i + 1]; del args[i:i + 2]; return v
    return d
WHEN = opt("--when", "reply"); SCALE = float(opt("--scale", 1)); N = int(opt("--n", 120))
PROMPTS = args or ["Who are you?", "Tell me about yourself.", "Describe your morning.", "What's on your mind right now?",
                   "What do you do all day?", "What matters most to you?", "Write a short poem about the sea.", "What is 12 times 8?"]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
M = "Qwen/Qwen3-4B"; tok = AutoTokenizer.from_pretrained(M)
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
ED = {}
for part in spec.split(","):
    L, n, x = part.split(":"); ED.setdefault(int(L), []).append((int(n), float(x.lstrip("+")) * SCALE))
FADE = int(WHEN[4:]) if WHEN.startswith("fade") else 0
st = {"t": 0}
def w(t): return 1.0 if WHEN in ("all", "reply") else max(0.0, 1 - t / FADE)
def hook(L):
    def f(mod, a):
        x = a[0]; T = x.shape[1]; x = x.clone()
        for n, d in ED[L]:
            if WHEN == "all": x[..., n] += d
            elif T > 1: x[0, -1, n] += d * w(0)          # prompt pass: only the last prompt position (it writes reply token 0)
            else: x[..., n] += d * w(st["t"])
        return (x,)
    return f
for L in ED: m.model.layers[L].mlp.down_proj.register_forward_pre_hook(hook(L))
out = []
for p in PROMPTS:
    ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to(DEV)
    if not torch.is_tensor(ids): ids = ids["input_ids"]
    past = None; cur = ids; toks = []; st["t"] = 0
    with torch.no_grad():
        for _ in range(N):
            o = m(cur, past_key_values=past, use_cache=True); past = o.past_key_values; st["t"] += 1
            nx = int(o.logits[0, -1].argmax()); toks.append(nx); cur = torch.tensor([[nx]], device=DEV)
            if nx in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")): break
    r = tok.decode(toks, skip_special_tokens=True).strip(); out.append({"prompt": p, "reply": r})
    print(f"\n### {p}\n{r}", flush=True)
(HERE / "out_Qwen3-4B" / f"demo_{time.strftime('%H%M%S')}.json").write_text(json.dumps({"spec": spec, "when": WHEN, "scale": SCALE, "out": out}, indent=1))
