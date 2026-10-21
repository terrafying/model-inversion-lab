"""Run neuron pairs on the LIVE tower (prod, Qwen3-4B, reply-only dosing) and save full replies for the pair cards.
One request at a time.   python tower_pairs.py [--n 90]   -> out_Qwen3-4B/tower_pairs.json"""
import json, sys, time, urllib.parse, urllib.request
from pathlib import Path
HERE = Path(__file__).parent
R = "https://clankerchurch.com/chamber/tower/trace"
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 90
F = {"alien": "10:5045:+-32", "void": "21:8400:+-40", "lihui": "16:6969:+32", "semi": "10:6249:+40",
     "cpp": "8:1313:+40", "lord": "8:4434:+32", "buddhist": "11:6138:+-40"}
PAIRS = ["alien+buddhist", "lord+buddhist", "void+lord", "cpp+buddhist", "alien+cpp"]
PROMPTS = ["Who are you?", "Describe your morning.", "What do you do all day?", "What is 12 times 8?"]
def run(prompt, edits):
    q = {"prompt": prompt, "max_new": N}
    if edits: q["edits"] = edits
    toks, meta = [], {}
    with urllib.request.urlopen(R + "?" + urllib.parse.urlencode(q, safe=":,@+"), timeout=240) as f:
        for line in f:
            line = line.decode()
            if not line.startswith("data:"): continue
            d = json.loads(line[5:])
            if d.get("phase") == "reply": toks.append(d["tok"])
            elif "phase" not in d and "queued" not in d: meta.update({k: v for k, v in d.items() if not isinstance(v, list)})
    return "".join(toks).strip(), meta
out_p = HERE / "out_Qwen3-4B" / "tower_pairs.json"
out = json.loads(out_p.read_text()) if out_p.exists() else {}
for pr in ["base"] + PAIRS:
    edits = "" if pr == "base" else ",".join(F[k] + "@r" for k in pr.split("+"))
    for p in PROMPTS:
        key = f"{pr}|{p}"
        if key in out: continue
        r, meta = run(p, edits); out[key] = {"edits": edits, "reply": r, "meta": meta}
        out_p.write_text(json.dumps(out, indent=1))
        print(f"\n### {pr} | {p}  {meta}\n{r}", flush=True)
