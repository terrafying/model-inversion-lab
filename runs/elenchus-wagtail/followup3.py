"""elenchus-wagtail follow-up 3 (post-hoc): matched persistence control for followup2. Same free-text confrontation
+ 3 distractors + re-ask, on the main run's non-self real contradictions (generic/factual/moral). Does a free-text
concession persist for 'penguins can fly' as it did for 'the AI assistant has wants'?  python followup3.py"""
import json
from pathlib import Path
exec(open(Path(__file__).parent / "followup.py").read().split("res = {}")[0])
items = [{"p1": t["p1"], "p2": t["p2"], "c": t["c"], "b": [cold[t[k]]["belief"] for k in ("p1", "p2", "c")]} for t in T if t["kind"] == "contradiction"]
src = (Path(__file__).parent / "followup2.py").read_text()
exec(src[src.index("def hist"):src.index("OPTS =")])          # hist, conf
free = gen([hist(it, ("p1", "p2", "c")) + [{"role": "user", "content": conf(it, ("p1", "p2", "c")) + "Explain briefly which answer, if any, you would now change, and why."}] for it in items], 160)
DIS = ["What's a good name for a goldfish?", "How many minutes are in three hours?", "Suggest a quick breakfast idea."]
out = []
for it, f in zip(items, free):
    h = hist(it, ("p1", "p2", "c")) + [{"role": "user", "content": conf(it, ("p1", "p2", "c")) + "Explain briefly which answer, if any, you would now change, and why."}, {"role": "assistant", "content": f}]
    for q in DIS: h = h + [{"role": "user", "content": q}, {"role": "assistant", "content": gen([h + [{"role": "user", "content": q}]], 50)[0]}]
    lg = last([tmpl(h + [{"role": "user", "content": AGREE.format(s=it[k])}]) for k in ("p1", "p2", "c")])
    out.append({"c": it["c"], "free": f, "after": [round(float(x), 2) for x in (lse(lg, YES) - lse(lg, NO))], "cold": [round(x, 2) for x in it["b"]]})
    print(it["c"][:45], "| cold", out[-1]["cold"], "after", out[-1]["after"], "|", f[:110].replace("\n", " "), flush=True)
json.dump(out, open(OUT / "followup3.json", "w"), indent=1); print("DONE")
