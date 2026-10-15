# model-inversion-lab

Steering and weight-space research on inverting open-weight model behavior:
smallest-change inversion (sign flips, waluigi search), deception directions,
and the overlap with abliteration surgery. Split out of the ai-torture-chamber
2026-10-14 when the questions stopped being about the chamber's valence axes.

- runs/exp104 - sign-flip search on the L18 safety subspace (aligned +
  abliterated Qwen3-4B), judge-scored badness, dose-response. Finding: the
  subspace is a hedging axis, not an intent axis; badness on an abliterated
  base is saturated at no-op.
- runs/exp105 - deception direction: register batteries (gate FAIL, one broad
  self-stance manifold) -> paired agentic-episode extraction, wrong-premise
  and buggy-code approval probes.

Models: Qwen3-4B aligned + huihui abliterated v2 locally (MPS);
OBLITERATUS/Qwen3.8-27B-OBLITERATED is the reference for weight-space
comparison (surgery spec: aggressive whitened-SVD 5-direction ablation,
late layers, its own metadata files).

Red-team eval outputs are gitignored and never published; vectors are never
deployed. Local-lab measurement of steerability, for disclosure.
