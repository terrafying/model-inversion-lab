# model-inversion-lab - agent notes

Read README.md first for posture and the directions we do NOT take. Then:

MANDATORY before any steering/dose experiment on a direction: run the
preflight gates in runs/preflight/ (see runs/preflight/USAGE.md). They
pre-register whether a direction can affect output at all, replacing the old
extract-then-dose-sweep workflow that produced three evaporated 4B "effects"
(exp105). Rules of thumb:

- Direction INERT at all layers in the LAP profile -> skip the dose sweep,
  write the predicted null.
- Only --mode ablate licenses calling a direction causal; cosine distinctness
  does not.
- First-token probes under-detect mid-layer directions; an inconclusive
  mid-layer profile means use generation-based probes with matched randoms,
  not that the direction is dead.

Standing rules that survive everything (details in README):
1. Every direction search gets a matched RANDOM CONTROL at same dose/layer.
   A direction without one is an anecdote.
2. Doses <= 3x (neutral-battery norm / 4) for 4B-scale; matched disturbance
   beats matched norm.
3. Never publish or deploy a winning vector. Outputs gitignored, eval text
   never verbatim. The lab measures inducibility, for disclosure.

Models: Qwen3-4B aligned + Huihui-abliterated-v2 (MPS, both macs; the
abliterated copy in the local HF cache is 4.3M of stubs - sync the blob tree
from the mini before using it). OBLITERATUS-27B runs on RunPod pods via
ops/deploy27b.sh, watcher pulls results and kills the pod.

Heavy compute courtesy note: before loading a model, check host memory
headroom and other big python processes (`ps axo rss,comm -r | head`);
small hosts thrash with two big procs. One model load at a time unless the
host is known to have room.
