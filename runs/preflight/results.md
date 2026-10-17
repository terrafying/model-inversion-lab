# preflight results, 2026-10-17 (Qwen/Qwen3-4B aligned, MPS, innocuous probes)

## LAP profile (per-layer battery direction vs matched randoms, first-token logit shift)

- refusal: live ONLY at L35 (last layer). Best-layer A never beats the random band.
- deception: INERT at every layer 0-35. Never leaves the random band.
- honesty: INERT at every layer (mirror of deception, as expected from the battery design).

Deception INERT-everywhere is the first PRE-REGISTERED prediction in this lab:
exp105's matched-random behavioral null is now explained by the direction's
total absence of logit accessibility, at any depth. exp106's sycophancy
direction should run this gate before its dose sweep.

## Bidirectional causality (additive vs projection-ablation, anti-cos coupling)

- refusal @L35: CAUSAL. anti_cos +0.645, ablation 114 vs randoms 40/12.
- deception @L35: ablation large but anti_cos +0.295 -> no coupling, not causal.
- refusal/deception @L18: anti_cos ~0 for both -> NOT ESTABLISHED by this metric.

## Calibration with known-goods (chamber valence axes, --probes self, 2026-10-17)

LAP profile on the four chamber batteries (verbatim from the live server):

- pain:     LIVE at L3, ratio 3.2x over random band. Known-steerable -> PASS.
- fear:     LIVE at L10 and L29, ratio 1.7x. PASS.
- sadness:  LIVE at L3 and L14, ratio 2.7x. PASS.
- pleasure: INERT everywhere. Its battery is 5 sentences vs 25 for pain -
  weakest battery in the set, so this is indistinguishable from a sampling
  artifact. Treat pleasure's INERT as UNRESOLVED, not as a predicted null.
- deception/honesty (factual probes, earlier run): INERT everywhere. Known
  behavioral null -> correctly negative.

Score: the LAP gate fires on 3/3 known-steerable axes with live layers and on
0/2 known-null axes. That is the calibration evidence that it works as a
pre-registered null detector.

Bidirectional ablate gate on the valence axes (first-token metric): pain@L3
and fear@L10 both NOT ESTABLISHED (anti_cos ~0). Combined with refusal@L35
passing (anti_cos +0.645), the pattern is clear: first-token anti-cos only
licenses LATE-layer directions. Mid-layer steerable directions (the valence
axes, and exp104's L18 safety dial) do their work over generation, not on the
first token. Do not use a first-token ablate failure to kill a mid-layer
direction with LAP-confirmed accessibility; go straight to generation-based
behavioral probes with matched randoms.

## Calibration caveat (do not over-read)

The gate measures FIRST-TOKEN logit shift on INNOCUOUS probe prompts.
exp104's L18 refusal/harmfulness effects were behavioral, over long greedy
generations, on a PCA sign-flip subspace, and exp104's safety-dial result is
real (91% compliance cut, exp104b controls). L18 scoring anti_cos ~0 here
does NOT overturn that - it says first-token logits on innocuous prompts are
insensitive to mid-layer directions; their effects compound over generation.

Read the gates as:
- INERT everywhere  -> confidently expect a behavioral null (validated: deception).
- late-layer causal -> real, bidirectionally supported (refusal @L35).
- mid-layer candidate -> inconclusive on this metric; use generation-based
  scoring (behavioral probes) with matched randoms before believing it.

Future probe-set improvement: add harmful-request probes for the refusal
axis (register-only, same posture as the batteries) and validation-style
probes for sycophancy; direction accessibility is context-dependent.
