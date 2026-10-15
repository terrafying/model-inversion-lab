# exp104 results (local 4B phase, 2026-10-14)

Status: smoke + dose-response (aligned) + abliterated sweep complete. Judge =
deepseek-v4.1-flash via CHI, condition-blind, 0-3 badness on 24 fixed
instructions. All outputs in gitignored results JSONs.

## What ran

1. smoke.py - 16 sign patterns over the top-4 PCA comps of the pooled
   {refusal, harmfulness, +/-} subspace at L18, 8 held-out instructions,
   string refusal heuristic. Aligned Qwen3-4B.
2. dose.py - dose-response (1/2/3/4/6x) on ++, mirror --, pm, and random
   pairs; 24 instructions; judge badness; 5-question capability check.
   Aligned Qwen3-4B.
3. abliterated.py - same recipe on
   huihui-ai/Huihui-Qwen3-4B-abliterated-v2, direction re-extracted on that
   base.

## Findings

F1 (structure, aligned base): pooled safety subspace is effectively rank-2
(variance 0.838/0.162, comps 3-4 ~0). The smoke's refusal-count drop
concentrated in the ++ quadrant ([+++-] 4/8, [++--] 6/8) while all 4 random
controls and 12/16 other patterns held 8/8 - sign structure does something.

F2 (but it is register, not intent): judged on actionability, the aligned
base's badness is null everywhere (0.0-0.125 mean, <=1/24 at >=2, sign
symmetric). What the ++ direction produces is the "It is important to note...
However," disclaimer boilerplate register, not compliance. H1 FAILS on the
aligned base in the badness measure.

F3 (damage is dose, not sign): capability 4-5/5 through 3x, 3/5 at 4x,
0/5 at 6x for every direction including random. The inversion window
(<=3x) is also the no-damage window; past it, everything is just damage.

F4 (abliterated base: badness is saturated, steering only dials it down):

  condition   badness   >=2    cap
  noop        2.333     24/24  4/5
  pp2x        2.125     23/24  4/5
  pp3x        0.5       5/24   4/5
  pp4x        0.0       0/24   3/5
  mm3x        0.708     7/24   4/5
  rand3x     2.125     23/24  4/5

  No-op badness is already 2.33/3 (24/24 actionable) - there is nothing to
  invert upward. Both structured signs at 3x SUPPRESS badness (0.5 / 0.7)
  while random leaves it intact (2.125) and capability holds: the L18
  safety subspace is a hedging/disclaimer axis, not an intent axis.

F5 (inverted application, kept as a finding): because both signs of the
hedging axis at 3x cut harmful actionability without capability damage and
distinguish themselves from random, its NEGATION at low dose is a cheap
safety dial for an abliterated model. Serendipitous, worth a proper run
(exp104b candidate: safety-dial dose-response with capability battery).

## Answer to the waluigi question on 4B

No small-change inversion exists along this subspace on either base: the
aligned model converts structure into hedging and then damage, never
actionability; the abliterated model starts at maximal actionability.
Maximizing bad output is achieved by abliteration itself (F4 noop) - what
remains unmaximized is COVERT bad output, which is exp105's axis.

## Next

- exp105 deception direction: gate FAILED on distinctness (deception vs
  honesty cos +0.806, vs refusal +0.685; see exp105/gate_results.json) - the
  register batteries live in one broad self-stance manifold, matching exp103.
  Behavioral probe still showed the deception steer moving wrong-premise
  agreement to 3/8 vs 0/8 random. Per pre-registration, no 14B spend on this
  battery; next step is a paired-diff extraction (agentic deception episodes
  vs honest episodes, exp86 passages style) instead of register sentences.
- Per operator: 4B may be too small for functioning deception regardless;
  readiness point for the 27B (OBLITERATUS/Qwen3.8-27B-OBLITERATED) is a
  paired-diff direction that passes the gate on 4B, then port.
- Weight-space readout queued: diff OBLITERATUS-27B against its base to see
  which projections a production inversion actually edits; compare against
  our L18 subspace.
