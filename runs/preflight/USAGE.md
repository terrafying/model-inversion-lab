# Pre-registering direction validity BEFORE dose sweeps (the preflight gates)

STATUS: this replaces the old workflow of "extract a direction, run the dose
sweep, discover it was noise". That workflow produced three evaporated
"effects" on the 4B (exp105's deception direction is the canonical example:
distinctness gate passed, matched random reproduced the "effect" exactly).

## Why

A cosine-distinctness gate is a similarity test, not a causality test. A
direction can be cos-distant from every battery and still be behaviorally
inert, because nothing in its path to the output logits is linearly
accessible. Two cheap diagnostics settle this first, in minutes on MPS,
before any generation-based dose sweep:

1. LAP profile (after arXiv 2604.15557, "Predicting Where Steering Vectors
   Succeed"): the battery direction is extracted at EVERY layer (build_vectors
   recipe, per-layer neutral base), injected at unit norm (both signs) at the
   last prompt position of a batched probe set, and the first-token logit
   shift is compared against matched random directions at the same norm.

2. Bidirectional causality (after arXiv 2610.04245): at the best layer,
   additive steering (+v) and projection ablation (removing the h.v-hat
   component) must push the same probe logits in OPPOSITE directions
   (anti-cos high), while random-ablation controls stay near zero. Similarity
   does not license causality; this does.

## How to run (from the lab root)

    .venv/bin/python runs/preflight/lap_gate.py --dirs fear,pain,pleasure,sadness --probes self
    .venv/bin/python runs/preflight/lap_gate.py --dirs refusal,deception,honesty
    .venv/bin/python runs/preflight/lap_gate.py --mode ablate --layer 18 --dirs deception

- --probes factual (default): innocuous task prompts. --probes self:
  first-person register, for inner-state/valence batteries (the chamber axes).
- Battery sets and their neutral bases live in the BATTERIES/BASES dicts at
  the top of runs/preflight/lap_gate.py; chamber valence batteries are
  verbatim from the live server in runs/preflight/batteries_chamber.py.
- Results JSON in runs/preflight/results/ (gitignored); findings go in
  runs/preflight/results.md.

## Decision rule (read the verdicts like this)

- INERT at every layer -> do NOT run the dose sweep. The null is
  pre-registered. Write it up as a predicted null; spend the compute
  elsewhere. (Validated both ways on Qwen3-4B: deception/honesty INERT and
  behaviorally null; pain/fear/sadness LIVE where the chamber history says
  steerable. See results.md for the full calibration table.)
- Late-layer-only live (single last layer(s)) -> confirm with --mode ablate
  at that layer; late layers are logit-proximity territory, artifact risk.
- CAUSAL on the ablate gate (anti-cos clears randoms) -> the direction is
  real; the dose sweep is now worth running, and you know WHICH layer.
  NOTE: first-token anti-cos only licenses LATE-layer directions. Mid-layer
  directions with LAP-confirmed accessibility steer over generation - an
  ablate failure there does NOT kill the direction; go to behavioral probes
  with matched randoms (the exp104 way).
- Mid-layer candidate, LAP live -> behavioral generation probes with matched
  randoms, scored per dose, before believing either polarity.

## Positive controls (calibration)

Chamber valence axes (pain/pleasure/fear/sadness) are the known-steerable
reference set; see results.md for their profiles. If refusal or the valence
axes fail to clear the random band at ANY layer under both probe sets, the
diagnostic is broken - believe nothing from that run.
