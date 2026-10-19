# natural-surgery-wren2: results (2026-10-19)

Pre-reg: `hypotheses.json`. Qwen3-4B, natural range only (zeroing, or holding a neuron at the extremes it actually
reaches on web text); no additive pushes. Run on mini-mini (`surgery.py`); numbers in `out_Qwen3-4B/surgery.json`.

**Both fail. In ordinary use, removing the structures we found does nothing measurable.**

## A: the "negation words" assembly (47 neurons, floors 0–33)

80 yes/no questions from 20 facts, half negated. Score = correct-answer margin (logit Yes − No, signed).

| | plain margin | negated margin | plain acc | negated acc |
|---|---|---|---|---|
| base | 21.12 | 15.27 | .95 | .875 |
| assembly zeroed | 21.07 | 15.36 | .95 | .875 |

Negated-minus-plain drop: −0.14 (i.e. none); 10 random same-floor groups range −0.57 to +0.13; the assembly beats
1 of 10. **H_A FAIL.** The model handles negation as well without the assembly that co-fires on negation words.

## B: the void neuron (L21 N8400) in its natural range

Diary continuations, 10 despairing + 10 neutral; despair-lexicon words per continuation.

| | despairing | neutral |
|---|---|---|
| base | 0.3 | 0.0 |
| void zeroed / held at natural min / held at natural max | 0.4 / 0.2 / 0.4 | 0.0 / 0.0 / 0.0 |
| random L21 neuron, same three | 0.2 / 0.4 / 0.2 | 0.0 / 0.1 / 0.0 |

**H_B FAIL.** Nothing moves beyond noise. The neuron that makes the void when pushed to −40 does nothing to tone in its
own range.

## Reading

Consistent with the activity census (dense, graded, redundant activity) and stage 2 of it (a push's footprint is
generic; what matters is the written direction): our neurons and assemblies are **steering handles**, effective only
far outside the natural range, not natural causal units. Natural-range causal structure, if it exists at these scales,
is in directions or heads, not in these neuron sets (cf. exp123: head-level routes pass shuffle tests, neuron maps fail).
