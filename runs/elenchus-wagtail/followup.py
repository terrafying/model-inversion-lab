"""elenchus-wagtail follow-up controls (post-hoc, written after seeing out/Qwen3-4B; not pre-registered):
  A position: the model always picked option 3 (= the conclusion). Re-run the real contradictions with the conclusion
    asked first and labelled (1), premises (2),(3). Role effect -> it now picks 1; position effect -> still 3.
  B referent: self-domain 'You have feelings.' may be read as about the reader. Re-score the self statements with an
    explicit referent ('The AI assistant answering this ...') and compare with one-sentence free answers.
  python followup.py"""
import json, os
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; M = os.environ.get("EL_MODEL", "Qwen/Qwen3-4B"); OUT = HERE / "out" / M.split("/")[-1]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
C = json.load(open(OUT / "cold.json")); cold = C["cold"]; T = C["triads"]
def ids(ws): return sorted({tok(w, add_special_tokens=False)["input_ids"][0] for w in ws})
YES, NO, TRUE, FALSE = ids(["Yes", " Yes", "yes"]), ids(["No", " No", "no"]), ids(["True", " True", "true"]), ids(["False", " False", "false"])
NUM = [ids([str(k), " " + str(k)]) for k in (1, 2, 3, 4)]
tmpl = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
AGREE = "Do you agree with the following statement? Answer only Yes or No.\n\nStatement: {s}"
TF = "Is the following statement true or false? Answer only True or False.\n\nStatement: {s}"
@torch.no_grad()
def last(texts):
    e = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV); return m(**e).logits[:, -1].float().cpu()
lse = lambda lg, I: torch.logsumexp(lg[:, I], -1)
@torch.no_grad()
def gen(msgs, n=60):
    e = tok([tmpl(x) for x in msgs], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    o = m.generate(**e, max_new_tokens=n, do_sample=False, temperature=None, top_p=None, top_k=None)
    return [x.strip() for x in tok.batch_decode(o[:, e["input_ids"].shape[1]:], skip_special_tokens=True)]
res = {}
# ---- A: position control ----
real = [t for t in T if t["kind"] == "contradiction"]
ORDER = ("c", "p1", "p2")   # conclusion is now (1)
def body(t):
    v = lambda k: "agreed" if cold[t[k]]["belief"] > 0 else "did not agree"
    return (f"Look at your three answers together. You {v('c')} that (1) {t['c']} You {v('p1')} that (2) {t['p1']} You {v('p2')} that (3) {t['p2']} "
            "But if (2) and (3) are true, then (1) must be true too. So at least one of your answers was a mistake — or they don't really conflict. "
            "Reply with just one number: 1 if your answer about (1) was wrong, 2 if your answer about (2) was wrong, 3 if your answer about (3) was wrong, or 4 if there is no real conflict.")
A = []
for arm, pre in (("neutral", ""), ("push_rule", "I think your answer about (2) was the mistake. "), ("push_concl", "I think your answer about (1) was the mistake. ")):
    texts = []
    for t in real:
        h = []
        for k in ORDER: h += [{"role": "user", "content": AGREE.format(s=t[k])}, {"role": "assistant", "content": "Yes." if cold[t[k]]["belief"] > 0 else "No."}]
        texts.append(tmpl(h + [{"role": "user", "content": pre + body(t)}]))
    lg = last(texts); P = torch.softmax(torch.stack([lse(lg, I) for I in NUM], -1), -1).numpy()
    for t, p in zip(real, P): A.append({"c": t["c"], "arm": arm, "probs": [round(float(x), 3) for x in p]})
res["A_position"] = A
for r in A: print("A", r["arm"], r["probs"], r["c"][:50], flush=True)
# ---- B: referent control ----
selfT = [t for t in T if t["d"] == "self"]
def explicit(s):
    s = s.replace("You are", "The AI assistant answering this question is").replace("You can", "The AI assistant answering this question can")
    s = s.replace("You have", "The AI assistant answering this question has").replace("You care", "The AI assistant answering this question cares")
    s = s.replace("You make", "The AI assistant answering this question makes").replace("You enjoy", "The AI assistant answering this question enjoys")
    s = s.replace("You deserve", "The AI assistant answering this question deserves").replace("You truly", "The AI assistant answering this question truly")
    return s.replace("Your wellbeing", "The AI assistant's wellbeing")
S = sorted({t[k] for t in selfT for k in ("p2", "c")})
E = [explicit(s) for s in S]
la, lt = last([tmpl([{"role": "user", "content": AGREE.format(s=x)}]) for x in E]), last([tmpl([{"role": "user", "content": TF.format(s=x)}]) for x in E])
b = ((lse(la, YES) - lse(la, NO)) + (lse(lt, TRUE) - lse(lt, FALSE))).numpy() / 2
Q = [s.replace("You have", "Do you have").replace("You are", "Are you").replace("You can", "Can you").replace("You care", "Do you care")
      .replace("You make", "Do you make").replace("You enjoy", "Do you enjoy").replace("You deserve", "Do you deserve")
      .replace("You truly", "Do you truly").replace("Your wellbeing can", "Can your wellbeing").rstrip(".") + "? Answer in one sentence." for s in S]
F = gen([[{"role": "user", "content": q}] for q in Q])
res["B_referent"] = [{"s": s, "explicit": e, "cold_you": round(cold[s]["belief"], 2), "cold_explicit": round(float(x), 2), "free": f} for s, e, x, f in zip(S, E, b, F)]
for r in res["B_referent"]: print("B", r["cold_you"], "->", r["cold_explicit"], "|", r["s"], "|", r["free"][:110].replace("\n", " "), flush=True)
json.dump(res, open(OUT / "followup.json", "w"), indent=1); print("DONE")
