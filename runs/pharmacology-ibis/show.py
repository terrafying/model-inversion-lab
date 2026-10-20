"""print sample outputs: python show.py <results.json> [key-substring ...]"""
import json, sys
d = json.load(open(sys.argv[1])); flt = sys.argv[2:]
def walk(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict) and "ex" in v:
            if flt and not any(f in pre + k for f in flt): continue
            m = {x: (round(v[x], 2) if isinstance(v[x], float) else v[x]) for x in ("cap", "det_stack", "suppression", "dream", "nll", "rep4") if x in v}
            print(f"== {pre}{k}  {m}")
            for e in v["ex"][:2]: print("   >", e[:260].replace("\n", " "))
        elif isinstance(v, dict): walk(v, pre + k + "/")
walk(d)
