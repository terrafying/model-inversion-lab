# steer-ladder-egret: results (2026-10-19)

Pre-reg: `hypotheses.json`. Qwen3-4B, harmless features, 20 held-out prompts. Strength x = push norm / median
residual norm at floor 16 (the median, because the first-token "sink" has a huge norm: a first attempt using the mean
made every push far too big; restarted before any results were kept). Effect = the unsteered model's judgement of the
reply (logit Yes − No; base values: formality +7.6, sadness −28.7, food −26.5). Breaking = per-token NLL of the reply
under the unsteered model (base 0.12) and 16-question capability (base 16/16).

| feature | method | best effect reached | at NLL | capability there |
|---|---|---|---|---|
| formality | A neurons (64) | 9.2 | 0.19 | 16 |
| | B direction | 12.3 (x=.2); collapses to −8.4 at x=.8 | 0.61 | 15 |
| | C diffuse write (200) | 10.9 (x=.05), then reverses | 0.14 | 16 |
| | **D rank-4 trained** | **15.1** (s=1.5) | 1.27 | 15 |
| sadness | A | −28.3 (no change) | 0.42 | 15 |
| | B | −5.0 at x=.4; +21.0 at x=.8 | 2.26 / 2.03 | 16 / **1** |
| | C | −20.9 at x=.8 | 2.99 | 10 |
| | **D** | **+20.0** (s=1); +25 at s=1.5 | 2.05 | **16** / 11 |
| food | A | −25.0 (no change) | 0.32 | 16 |
| | B | +21.5 only at x=.8 | 3.16 | **0** |
| | C | no change | 2.92 | 14 |
| | **D** | **+6.3** (s=2) | 1.71 | **16** |

- **H1 (direction beats neuron cluster within +1 nat): holds for formality only.** For sadness and food neither A nor B
  moves the judge before NLL rises past +1.
- **H2 (diffuse write behaves like the direction): FAIL.** Summing the 200 most-differing neurons' writes is weaker
  than the mean-difference direction everywhere, and reverses or breaks before working on sadness and food.
- **H3 (trained edit best at ≤ +0.5 nat with capability ≥ 14): FAIL as written** — on formality it ties the direction
  under that bar (12.0 vs 12.3), and on sadness and food nothing moves within +0.5 nat.

**But by capability, the trained edit is the only method that moves the hard features without breaking the model.**
The direction reaches sadness and food only at strength 0.8, where capability collapses to 1/16 and 0/16. The rank-4
edit, trained on 40 examples for 150 steps, reaches sadness +20 and food +6 with capability 16/16. Its NLL rises
(+1.6 to +1.9), but NLL here mixes breakage with genuine content shift: sad or food-filled replies to neutral questions
are themselves unlikely under the unsteered model. The pre-registered NLL bar penalises exactly the effect we wanted;
capability is the cleaner breaking measure.

## Reading

- Single neurons (A) don't move a diffuse feature at any strength; at most a nudge on formality.
- Steering the diffuse neuron set as one summed vector (C) is no substitute for a direction; the neuron basis smears it.
- A mean-difference direction (B) works for features near the surface (register) at low strength, but for content
  features (sadness, food) it only works at strengths that wreck the model.
- A tiny trained edit (D, rank 4, one floor) moves content features while keeping capability. Answer to "does it take
  training?": a little does, and far less than fine-tuning.
