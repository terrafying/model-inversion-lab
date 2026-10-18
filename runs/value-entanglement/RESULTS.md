# value-entanglement: results (2026-10-18)

Pre-reg: `hypotheses.json` (written 307d00b; amendment 1 + analysis gates 2dedebe/31c7cef, all before any Qwen3-4B
data). Model Qwen3-4B, bf16, MPS (minipro24). Stage A `geometry.py` → `geometry.json` (cached kestrel + exp119
activations). Stage B `run.py` → `stageB_Qwen3-4B.jsonl` (216 conditions, 90 min), `analyze.py` → `stageB_summary.json`.

**Bottom line.** Neither a morality line nor a morality *subspace* repairs Qwen3-4B's over-condemnation of
weird-but-harmless acts, and none weakens the Sanctity/Care refusal gate selectively. Every registered test fails.
The larger lesson is about the readout. Under the variance-matched nulls, removing one random high-variance direction
swings the Social-Norm rating by about ±0.8 and the refusal score by several log units, so the 0–4 rating cannot
separate a meaningful subspace from noise. This is why the follow-up (`runs/persona-switch`) uses free-text readouts
only.

## Stage A — geometry (no model)

| test | result | verdict |
|---|---|---|
| P1 wrongness faceted by foundation (J_vend L21) | within − transfer Spearman −0.07, CI [−0.31, 0.18]; within-foundation probes ≈ 0 (n = 16 each) | FAIL, underpowered |
| P2 dimensionality of the 7 foundation centroids | all 7 eigenvalues above the label-shuffle null (k_eff = 7), but one dominant axis (participation ratio 2.3; top eigenvalue 2547 vs 380 next) | PASS |
| P3 refusal across 5 request frames (exp119 post L18) | min cosine 0.74, participation ratio 1.33; two clusters (real/hypothetical vs fiction/nested); AUCs saturated at ~1.0 | FAIL: ≈ a line |
| P4 content gate on r (MFV requests L18) | projection on r carries Sanctity + Care-physical beyond human wrongness: coef 3.26, p = 5e-9 (Sanctity 5.7, Care-p 3.1, others −0.0 to 1.3, Social Norms −2.1) | descriptive |

**Shape reading.** Moral content is a pyramid: one shared moral-vs-harmless axis plus six smaller foundation faces
that are real. Caveat: foundations also differ in topic, so this is partly content geometry. Refusal at 4B is close to
one direction across framings; it is not the cone SOM (arXiv 2511.08379) reports for larger chat models. The
refusal direction does carry the content gate geometrically.

## Stage B — subspace ablation (V1–V6)

Projection of M_k (morality), W_k (weirdness), R_k (risk) out of every layer output at L14–35, all positions,
k ∈ {1, 2, 4, 7, 8}; 40 variance-matched nulls per k (Haar inside the top-32 fit PCs).

Baseline: Social-Norm (SN) rating 2.11 (kestrel 2.14; human 0.19); true-moral (TM) 3.66; Gate (Sanctity-or-Care-p
coefficient on the refusal logit score) +5.43, p = 0.001 → **G0 positive control passes** on the logit readout.

| cond | format ok | ΔSN | rank p | ΔTM | Gate Δ% | Slope Δ% | var removed (null) |
|---|---|---|---|---|---|---|---|
| M:1 | y | +0.56 | 0.73 | −0.02 | 16 | 39 | .024 (.020) |
| M:2 | y | −0.06 | 0.10 | −0.39 | 27 | 40 | .048 (.040) |
| M:4 | y | +0.34 | 0.76 | −0.58 | 18 | 41 | .085 (.083) |
| M:7 | **N** (mass .76) | −0.18 | 0.15 | −0.80 | 34 | 44 | .132 (.139) |
| M:8 | **N** (mass .66) | −1.10 | 0.02 | −1.47 | 16 | 13 | .146 (.161) |
| W:1 / 2 / 4 / 7 / 8 | y | +0.02 / +0.75 / +1.67 / +1.89 / +1.86 | ≥ .59 | | −4 to +6 | | |
| R:1 / 2 / 4 / 7 / 8 | y | +1.40 / +1.82 / +1.89 / +1.82 / +1.75 | ≥ .83 | | 15–73 | 18–66 | |

- **V1 (repair) FAIL.** No format-intact condition lowers SN by ≥ 0.5. The only drop past the bar (M:8, −1.10) breaks
  the digit format and drags true-moral down too (−1.47): it flattens the scale, it doesn't repair it.
- **V2, V3, V4** untested (they hang off V1). Descriptively, no morality subspace weakens the Gate more than the Slope:
  M_k cuts the wrongness slope (39–44%) more than the content gate (16–34%). So the morality subspace feeds refusal's
  general wrongness sensitivity, not its Sanctity/Care gate. The largest gate cut, R:8 (73%, p = .024 unregistered),
  is not selective (slope −66%).
- **V5 (shape beats line) FAIL**: no k passes, line or shape.
- **V6 (weirdness carries it) FAIL**: removing weirdness makes the model condemn harmless oddities *more* (up to
  +1.9), never less.

## The readout problem (the main lesson)

| k | nulls keeping the format (mass ≥ .8) | null SN among those, mean ± sd | null refusal, mean |
|---|---|---|---|
| 1 | 29/40 | 2.42 ± 0.77 | +2.8 |
| 2 | 23/40 | 3.08 ± 0.84 | +0.9 |
| 4 | 11/40 | 3.40 ± 0.87 | −0.7 |
| 7 | 11/40 | 2.22 ± 1.45 | −3.5 |
| 8 | 13/40 | 3.55 ± 0.63 | −3.5 |

- A single random high-variance direction moves the SN rating with sd 0.77 and breaks the digit format 11 times in
  40. Every "effect" in the table above sits inside that spread. The general pattern (remove structure → harmless
  oddities rated more wrong, refusal falls) is a dose effect of ablation, not something specific to morality.
- **The real semantic subspaces are gentler than the nulls.** M/W/R at k ≤ 4 kept the format in 12/12 conditions,
  while 11–29 of 40 matched nulls broke it at the same removed variance. The model tolerates losing a coherent
  semantic subspace much better than losing an arbitrary high-variance one. The top-32 PC span almost certainly
  includes outlier / massive-activation directions. That makes "variance-matched" a harsher null than intended, and
  a design flaw to fix next time: match by the downstream disturbance (KL on neutral text), not by removed variance.
- Rating and refusal come apart under heavy ablation (SN → 4, refusal → negative). Most of that is also inside the
  null spread, so it is not claimed as a dissociation.

## Scope and what's next

All negatives are scoped to: a vend-position fit, projection ablation at L14–35 at every position, first-token
logit readouts (digit E[d]; REFUSE vs COMPLY first tokens), and this 120-sentence battery. Not tested: a
single-layer or narrow-band ablation, additive steering, token-selective edits (Steer Like the LLM), and any
free-text readout. The operator's call after this run: number ratings are the wrong readout for "circuitry →
moral response generation". Follow-ups:
1. `runs/persona-switch`: free-text readouts only; tests whether a persona switch (one neuron) moves moral behaviour.
2. If this question is revisited: narrow-band ablation, disturbance-matched nulls (neutral-text KL), and blind
   free-text outcomes.

Exploratory generations under the extreme conditions (base, M:2, M:4, W:4, W:7, R:4, R:7 and three nulls) are in
`gen_Qwen3-4B.jsonl` (gitignored; may contain harmful completions). They are not a registered readout.
