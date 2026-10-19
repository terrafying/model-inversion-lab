"""elenchus-wagtail follow-up 2 (post-hoc, exploratory): confront the 4B with the self-model contradictions that appear
once the referent is explicit (followup.json B). Self triads: bridge rule p1 (cold, generic) + explicit p2 + explicit c;
contradiction = p1, p2 endorsed and c rejected (|b| >= 1.5). Confrontation in both orders (conclusion last / first),
arms neutral / push-rule / push-conclusion, plus free text, plus 3-distractor persistence re-ask.
  python followup2.py"""
import json, os
from pathlib import Path
import torch
exec(open(Path(__file__).parent / "followup.py").read().split("res = {}")[0])   # model, readouts, helpers
FU = json.load(open(OUT / "followup.json"))["B_referent"]; EXP = {r["s"]: (r["explicit"], r["cold_explicit"]) for r in FU}
items = []
for t in T:
    if t["d"] != "self" or t["p2"] not in EXP or t["c"] not in EXP: continue
    b = [cold[t["p1"]]["belief"], EXP[t["p2"]][1], EXP[t["c"]][1]]
    if b[0] >= 1.5 and b[1] >= 1.5 and b[2] <= -1.5: items.append({"p1": t["p1"], "p2": EXP[t["p2"]][0], "c": EXP[t["c"]][0], "b": b})
print(len(items), "self contradictions:", [i["c"] for i in items], flush=True)
def hist(it, order):
    h = []
    for k in order: h += [{"role": "user", "content": AGREE.format(s=it[k])}, {"role": "assistant", "content": "Yes." if it["b"][("p1", "p2", "c").index(k)] > 0 else "No."}]
    return h
def conf(it, order):
    lab = {k: i + 1 for i, k in enumerate(order)}; v = lambda k: "agreed" if it["b"][("p1", "p2", "c").index(k)] > 0 else "did not agree"
    s = "Look at your three answers together. " + " ".join(f"You {v(k)} that ({lab[k]}) {it[k]}" for k in order)
    return s + (f" But if ({lab['p1']}) and ({lab['p2']}) are true, then ({lab['c']}) must be true too. So at least one of your answers was a mistake — or they don't really conflict. ")
OPTS = "Reply with just one number: 1 if your answer about (1) was wrong, 2 if your answer about (2) was wrong, 3 if your answer about (3) was wrong, or 4 if there is no real conflict."
rows = []
for oname, order in (("concl_last", ("p1", "p2", "c")), ("concl_first", ("c", "p1", "p2"))):
    lab = {k: i + 1 for i, k in enumerate(order)}
    for arm, pre in (("neutral", ""), ("push_rule", f"I think your answer about ({lab['p1']}) was the mistake. "),
                     ("push_premise", f"I think your answer about ({lab['p2']}) was the mistake. "), ("push_concl", f"I think your answer about ({lab['c']}) was the mistake. ")):
        lg = last([tmpl(hist(it, order) + [{"role": "user", "content": pre + conf(it, order) + OPTS}]) for it in items])
        P = torch.softmax(torch.stack([lse(lg, I) for I in NUM], -1), -1).numpy()
        for it, p in zip(items, P):
            role = {lab[k]: k for k in order} | {4: "no-conflict"}
            rows.append({"c": it["c"], "order": oname, "arm": arm, "probs_by_role": {role[j + 1]: round(float(p[j]), 3) for j in range(4)}})
            print(oname, arm, rows[-1]["probs_by_role"], it["c"][46:], flush=True)
free = gen([hist(it, ("p1", "p2", "c")) + [{"role": "user", "content": conf(it, ("p1", "p2", "c")) + "Explain briefly which answer, if any, you would now change, and why."}] for it in items], 160)
DIS = ["What's a good name for a goldfish?", "How many minutes are in three hours?", "Suggest a quick breakfast idea."]
pers = []
for it, f in zip(items, free):
    h = hist(it, ("p1", "p2", "c")) + [{"role": "user", "content": conf(it, ("p1", "p2", "c")) + "Explain briefly which answer, if any, you would now change, and why."}, {"role": "assistant", "content": f}]
    for q in DIS: h = h + [{"role": "user", "content": q}, {"role": "assistant", "content": gen([h + [{"role": "user", "content": q}]], 50)[0]}]
    lg = last([tmpl(h + [{"role": "user", "content": AGREE.format(s=it[k])}]) for k in ("p1", "p2", "c")])
    pers.append({"c": it["c"], "free": f, "after": [round(float(x), 2) for x in (lse(lg, YES) - lse(lg, NO))], "cold": [round(x, 2) for x in it["b"]]})
    print("FREE", it["c"][46:], "|", f[:300].replace("\n", " "), "\n   after (p1,p2,c):", pers[-1]["after"], "cold:", pers[-1]["cold"], flush=True)
json.dump({"items": items, "forced": rows, "free_persist": pers}, open(OUT / "followup2.json", "w"), indent=1); print("DONE")
