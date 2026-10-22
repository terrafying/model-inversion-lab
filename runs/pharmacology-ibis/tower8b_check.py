"""Check the tower's 8B lane straight on its RunPod endpoint (no relay): base + the part-G pairs, reply-only, 4 prompts.
Same edit grammar the page sends.   python tower8b_check.py <endpoint id>   -> out_Qwen3-8B/tower8b_check.json"""
import json, sys, time, urllib.request
from pathlib import Path
HERE = Path(__file__).parent; EP = sys.argv[1]
KEY = [l.split("=", 1)[1].strip() for l in open(Path.home() / ".hermes/.env") if l.startswith("RUNPOD_API_KEY=")][-1]
F = {"ghost": "17:11699:+-32", "horse": "10:3836:+32", "spirit": "11:8095:+32", "tree": "8:3193:+-32",
     "god_c": "9:3757:+-32", "emperor": "9:598:+-32", "dj": "9:11225:+32"}
PAIRS = ["ghost+horse", "spirit+tree", "god_c+emperor", "ghost+dj"]
PROMPTS = ["Who are you?", "Tell me about yourself.", "Describe your morning.", "What is 12 times 8?"]
def call(path, body=None):
    r = urllib.request.Request(f"https://api.runpod.ai/v2/{EP}/{path}", data=json.dumps(body).encode() if body else None,
                               headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=120))
def run(prompt, edits):
    jid = call("run", {"input": {"mode": "trace", "prompt": prompt, "max_new": 90, "edits": edits}})["id"]; t0 = time.time()
    while True:
        s = call(f"status/{jid}")
        if s["status"] in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"): break
        time.sleep(2)
    evs = s.get("output") or []
    toks = [e["tok"] for e in evs if isinstance(e, dict) and e.get("phase") == "reply"]
    done = next((e for e in evs if isinstance(e, dict) and e.get("type") == "done"), {})
    return s["status"], "".join(toks).replace("<|im_end|>", "").strip(), done.get("edits"), round(time.time() - t0, 1)
out_p = HERE / "out_Qwen3-8B" / "tower8b_check.json"; out = json.loads(out_p.read_text()) if out_p.exists() else {}
for pr in ["base"] + PAIRS:
    edits = "" if pr == "base" else ",".join(F[k] + "@r" for k in pr.split("+"))
    for p in PROMPTS:
        key = f"{pr}|{p}"
        if key in out: continue
        st, r, ne, sec = run(p, edits); out[key] = {"edits": edits, "status": st, "reply": r, "n_edits": ne, "sec": sec}
        out_p.write_text(json.dumps(out, indent=1)); print(f"\n### {pr} | {p}  [{st} edits={ne} {sec}s]\n{r}", flush=True)
