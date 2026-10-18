# Research directions — what the 2026 literature opened up

Written 2026-10-18 after the literature sweep (`docs/related_work.md`) and the single-neuron replication
(`runs/refusal-neuron-stilt`). Ranked; each is pre-registerable on assets we already have (kestrel's 132 MFV
items + human means, exp119's harmful/benign twins, the J/YN/REV readouts, the r-dial, the verdict heads).

## 1. Value entanglement, taken further — the unifying bet  [PRE-REGISTERED: runs/value-entanglement]
Extends *Value Entanglement* (arXiv 2602.19101: moral value contaminates grammatical/economic judgment,
repaired by ablating a morality vector) onto our two threads.
- (a) Replicate the repair on our wrongness axis: project out one morality direction M; does Social-Norm
  (weird-but-harmless) over-condemnation drop toward human while true-moral wrongness stays intact?
- (c) The headline: is that SAME M the thing that content-gates refusal (kestrel: Sanctity+Care, p=.026)?
  If one ablation fixes the rating AND weakens the refusal gating, moral-map and refusal are one direction.
- (also) Map the entanglement graph (moral -> factual/risk/aesthetic) and test asymmetry (does moral
  dominate, or does risk bleed back?).
Cheap, forward-pass only, no pod. Hypotheses V1-V4 in runs/value-entanglement/hypotheses.json.

## 2. Polyhedral refusal — handle the cone, not the line
*SOM Directions* (arXiv 2511.08379): refusal is a multi-directional polyhedral cone, not one direction.
Reframes our lever/text gap — our r is a single handle on a k-D object, which is why single-direction edits
moved the score but never flipped generation.
- Build the cone: exp119's five frames each give a refusal direction; plus an SVD of harmful-benign
  activation differences. Do they span a low-dim cone (cosines, eigen-knee) or collapse to one line?
- Pre-reg: ablating the top-k refusal subspace flips generated refusal on >= N/24 harmful prompts where
  single-direction ablation flipped 0; matched random k-subspace nulls.
- Stack with *Steer Like the LLM* (arXiv 2605.03907): a TOKEN-SELECTIVE edit on the CONE is the strongest
  version — the two candidate fixes for the lever/text gap, combined.

## 3. Steer like a prompt — the lever/text gap as a method artifact
*Steer Like the LLM* (2605.03907): popular steering is position-blind (hits every token uniformly) where a
prompt is token-selective; prompt-shaped steering transfers to behaviour. Our edits were onset-only or
all-token. Test: a token-selective edit (strong on the question span, per deference-trace's L16-18 finding)
that finally moves free-text judgment where r/w/v/heads/single-neuron did not.

## 4. Moral verdict as a polytope whose faces are the foundations  [sleeper]
Combine SOM's cone idea with the moral-knowledge-structure work (2608.27402) + SAE moral features: is
wrongness not a scalar but a polytope with Care/Fairness/Sanctity/... as vertices? Connects verdict heads to
foundation geometry. Higher-risk, biggest structural payoff.

## Carried-over caveats
- The single-neuron bypass (2605.08513) did NOT reproduce on our readout at 4B or 8B (incl. their exact
  neuron L14 F7924) — at most it strips refusal boilerplate, never the decision. Clamp-value (benign mean vs
  their derived m*) is a confound; a clamp-value sweep would settle it. See runs/refusal-neuron-stilt/RESULTS.md.
- Every negative here is scoped to the method tested (the standing rule). Vary the intervention primitive and
  the clamp value before concluding "not reachable."
