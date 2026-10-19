"""Stage K (hypotheses.json): push two verified finds together. Each pair at both members' verified doses on the
20 prompts; classify BLEND / DOMINATE / CANCEL by each member's own readout; texts saved for site cards."""
import itertools, json
from pclib import HERE, gen, hits, push_of, readouts, rep4
OUT = HERE / "out_Qwen3-4B"; RO = readouts()
FINDS = ["alien", "void", "li-hui", "L7N4432", "L8N1313", "L7N8486", "L8N1263", "L10N7963"]
res = json.load(open(OUT / "pairs.json")) if (OUT / "pairs.json").exists() else {}
for a, b in itertools.combinations(FINDS, 2):
    key = f"{a}+{b}"
    if key in res: continue
    (La, na, sa, da, rxa, _), (Lb, nb, sb, db, rxb, _) = RO[a], RO[b]
    T = gen(push_of([(La, na, sa * da), (Lb, nb, sb * db)]))
    ha, hb = hits(T, rxa), hits(T, rxb); coh = sum(rep4(t) < .3 for t in T)
    cls = "BLEND" if ha >= 6 and hb >= 6 else ("DOMINATE:" + (a if ha > hb else b) if max(ha, hb) >= 6 and min(ha, hb) < 3 else ("CANCEL" if max(ha, hb) < 3 else "MIXED"))
    res[key] = {"a": a, "b": b, "hits_a": ha, "hits_b": hb, "coherent": coh, "class": cls, "texts": T}
    print(f"{key:22s} {a}:{ha:2d} {b}:{hb:2d} coherent {coh:2d}/20  {cls:18s} | {T[1][:110]}", flush=True)
    json.dump(res, open(OUT / "pairs.json", "w"), indent=1)
print("DONE")
