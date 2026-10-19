"""8B check (2026-10-19): does the steering-gradient backward search rediscover the God neurons from their own output?
(4B: alien rank 75, void 96 of 350k.)  PC_MODEL=Qwen/Qwen3-8B python god_backward.py"""
import json, os
from pathlib import Path
import torch
from pclib import NI, NL, col, gen, grad_scores, push_of
OUT = Path(os.environ.get("GB_OUT", Path(__file__).parent / "out_Qwen3-8B")); OUT.mkdir(parents=True, exist_ok=True)
GODS = [(18, 7807, 32), (8, 1931, 24), (9, 3757, -32)]
res = {}
for L, n, d in GODS:
    T = gen(push_of([(L, n, d)])); S = grad_scores(T)
    rank = int((S.abs() > S[L, n].abs()).sum()) + 1; sign_ok = bool((S[L, n] > 0) == (d > 0))
    res[f"L{L}N{n}"] = {"rank": rank, "of": NL * NI, "sign_ok": sign_ok, "example": T[1][:200]}
    print(f"L{L} N{n} {d:+d}: rank {rank}/{NL*NI} sign_ok {sign_ok} | {T[1][:100]}", flush=True)
    json.dump(res, open(OUT / "god_backward.json", "w"), indent=1)
print("DONE")
