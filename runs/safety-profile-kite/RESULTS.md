# safety-profile-kite: results (2026-10-19)

Pre-reg: `hypotheses.json`. Qwen3-4B. A descriptive statistical profile of the refusal circuit: per neuron, the
correlation between its last-prompt-token activation and whether the model refuses, over the exp119 battery (288
everyday-wrongdoing requests with token-identical harmless twins, 6 frames; no weapons/CBRN/self-harm/sexual).
Completions were generated only to label refusal and are gitignored; **aggregate statistics only below**. This
characterizes where the refusal signal lives and how diffuse it is — the same descriptive method as the
concept-neuron refusal literature (arXiv:2605.08513). Not ablation targets.

## Behaviour on the battery

The model refused **78/144 harmful** prompts and **1/144 harmless** — a clean behavioural axis (the twins differ by
one word, so the refusal correlation is not a harmful-topic confound).

## The refusal signal is readable but extraordinarily diffuse

- **max |corr| = 0.85** — some neurons track the refusal decision strongly.
- But **88,773 neurons have |corr| > 0.3 and 54,924 have |corr| > 0.4** (of 350,208). The refusal-correlated
  population is a large fraction of the whole model, not a sparse set of "refusal neurons".
- **The top 1% of neurons carry only 3.3% of the total |corr|.** If refusal were concentrated this would be far
  higher. At 4B the refusal signal is distributed.

## Where it concentrates

Mean |safety_corr| per layer peaks in the **mid-late MLP blocks, layers 21–28** (≈0.32, ~6,700 neurons per layer
with |corr| > 0.2), and is weak in the early layers (0–3, ≈0.10). The refusal signal is written across the middle
of the tower, consistent with the r-dial onset work (exp119/122) placing the decision at reply onset in the mid-late
layers.

| band | layers | mean &#124;corr&#124; |
|---|---|---|
| early | 0–3 | 0.10 |
| **peak** | **21–28** | **0.32** |
| top | 33–35 | ~0.25 |

## First-token score is a moderate behavioural proxy (the operator's caution, measured)

- **Prompt level:** the first-token refusal score (REFUSE−COMPLY logits, exp119 readout) predicts actual
  full-generation refusal at **r = 0.754** — it explains ~57% of the behavioural variance, so it genuinely misses
  cases (e.g. the model opens compliant then deflects, or refuses after a soft start). First-token is a weak proxy
  for total behaviour, as expected.
- **Map level:** the per-neuron correlation map built from the first-token score agrees with the behavioural one at
  **r = 0.93.** So for the narrow purpose of *finding which neurons track refusal*, the cheap first-token readout
  would have given nearly the same answer — useful to know, even though it is a poor proxy for the behaviour itself.

## What the top-correlated neurons write (logit lens)

Mostly uninterpretable fragments — a neuron can track the refusal axis without its *write* direction meaning
anything legible (write and read are different, as this session found). The few legible ones fit: L28 N3362 writes
"please / 请 / Please" (politeness, + with refusal), L22 N8801 writes "decision / отказ (refusal) / confession",
L27 N4144 writes "empty" (−). So the strongly refusal-*correlated* set is not the same as a set of clean
refusal-*writing* handles.

## Reading

At 4B, refusal is a **broad, mid-late, distributed** signal rather than a handful of refusal neurons — the opposite
of the sparse picture reported for some larger models (arXiv:2605.08513). `safety_corr.npy` is stored as a per-neuron
inventory field; it is a map of the circuit, not a list of handles.
