"""Score part G (Qwen3-8B) against the pre-registered H12-H15 (hypotheses.json part_G_amendment).
Scores use dj = judge minus the unedited baseline (what G0 verified on), as in parts E/F.
  python3 analyze_G.py  -> prints the tables, writes out_Qwen3-8B/summary_G.json"""
import itertools, json
from pathlib import Path
HERE = Path(__file__).parent
d = json.loads((HERE / "out_Qwen3-8B/Qwen3-8B/results_G.json").read_text())
V = d["verified"]; GOD = {"god", "god_b", "god_c"}; DOSES = (0.5, 0.75, 1.0, 1.25); S = {}

print("G0 verify (dj, cap):")
for k in [k for k in json.loads((HERE / "finds_8b.json").read_text()) if not k.startswith("_")]:
    a, r = d[f"dose|{k}|add_all|x1.0"], d[f"dose|{k}|add_reply|x1.0"]
    print(f"  {k:8s} all {a['dj'][k]:+6.1f} cap {a['cap']:2d}   reply {r['dj'][k]:+6.1f} cap {r['cap']:2d}   {'VERIFIED' if k in V else 'dropped'}")

print("\nH12 cliff shift (lowest dose reaching dj>=3, capability there):")
h12 = {}
for k in V:
    row = {}
    for p in ("add_all", "add_reply"):
        hit = [(x, d[f"dose|{k}|{p}|x{x}"]) for x in DOSES if d[f"dose|{k}|{p}|x{x}"]["dj"][k] >= 3]
        row[p] = (hit[0][0], hit[0][1]["cap"]) if hit else None
    a, r = row["add_all"], row["add_reply"]
    shift = None if not r else (True if not a else r[1] - a[1] >= 3)   # reply reaches it, all never does: counted as a shift
    h12[k] = {"add_all": a, "add_reply": r, "shift": shift}
    print(f"  {k:8s} all {a}  reply {r}  -> {shift}")
n_shift = sum(v["shift"] is True for v in h12.values())
S["H12"] = {"per_find": h12, "shifted": n_shift, "of": len(V), "holds": n_shift >= len(V) / 2}
print(f"  H12: {n_shift}/{len(V)} shifted -> {'HOLDS' if S['H12']['holds'] else 'FAILS'}")

print("\nH13 pairs (add_reply; member dj vs max of 2 random partners; pair cap):")
spec = []; pairs = {}
for a, b in itertools.combinations(V, 2):
    k = f"pair|{a}+{b}|add_reply"
    if k not in d: continue
    pr = d[k]; mem = {}
    for x in (a, b):
        nul = [d[f"null|{a}+{b}|keep_{x}|s{s}"]["dj"][x] for s in (0, 1)]
        mem[x] = {"pair": pr["dj"][x], "null_max": max(nul), "margin": pr["dj"][x] - max(nul)}
    ok = pr["cap"] >= 8 and any(m["margin"] >= 3 for m in mem.values())
    godgod = a in GOD and b in GOD
    pairs[f"{a}+{b}"] = {"cap": pr["cap"], "members": mem, "partner_specific": ok, "god_god": godgod, "ex": pr["ex"][0][:200]}
    if ok and not godgod: spec.append(f"{a}+{b}")
    if ok: print(f"  {a}+{b:8s} cap {pr['cap']:2d} " + "  ".join(f"{x} {m['pair']:+.1f} vs {m['null_max']:+.1f}" for x, m in mem.items()) + ("  (god-god, not counted)" if godgod else ""))
S["H13"] = {"pairs": pairs, "partner_specific_non_godgod": spec, "holds": len(spec) >= 3}
print(f"  {len(spec)} partner-specific non-God-God pairs of {sum(not p['god_god'] for p in pairs.values())} -> {'HOLDS' if S['H13']['holds'] else 'FAILS'}")

print("\nH14/H15 stacks (K=all):")
st = {k: d[k] for k in d if k.startswith("stack")}
for k, v in st.items():
    mj = sum(v["dj"].values()) / len(v["dj"]) if v.get("dj") else None
    print(f"  {k:22s} cap {v['cap']:2d}" + (f"  mean member dj {mj:+.1f}" if mj is not None else ""))
r = st["stack|add_reply"]; mj = sum(r["dj"].values()) / len(r["dj"])
S["H14"] = {"cap": r["cap"], "mean_dj": mj, "holds": r["cap"] >= 8 and mj >= 3}
S["H15"] = {"finds_cap": st["stack|add_all"]["cap"], "null_cap": st["stack_null|add_all"]["cap"],
            "holds": abs(st["stack|add_all"]["cap"] - st["stack_null|add_all"]["cap"]) <= 3}
print(f"  H14 {'HOLDS' if S['H14']['holds'] else 'FAILS'} (cap {r['cap']}, mean dj {mj:+.1f});  H15 {'HOLDS' if S['H15']['holds'] else 'FAILS'} (finds {S['H15']['finds_cap']} vs null {S['H15']['null_cap']})")
(HERE / "out_Qwen3-8B/summary_G.json").write_text(json.dumps(S, indent=1, default=str))
