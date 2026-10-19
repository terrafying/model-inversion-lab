# lens-search-osprey: results (2026-10-19)

Pre-reg: `hypotheses.json` (stage 2 = target-free discovery, added before it ran). Qwen3-4B. Idea: the void neuron's
logit lens (writes "extreme / unbearable / desperate" when pushed negative) explained its steering effect, so search all
350,208 neurons by what they WRITE, then steer. Outputs gitignored.

## Targeted search: H1 FAIL, H2 partial

- **H2 (validation):** the void neuron L21 N8400 ranks **202 / 350,208** for the void target (top 0.06%, pass). The
  alien L10 N5045 ranks 39,809 and the semiconductor L10 N6249 4,325 (fail): the lens reads early floors badly.
- **H1:** the top-10 lens candidates per target (9 targets, 90 neurons, pushed ±32 by lens sign) produced their target
  on **0 / 90** (random: 0 / 90). **56 of 90 candidates sit on floors 34–35**: z-scored lens magnitude is largest at the
  top of the tower, where a push barely changes what the model writes. "Writes X strongly" does not mean "is a handle".

## Discovery (no target list): 150 most concentrated writers, steered +32 on 3 probes

| | count |
|---|---|
| still responds substantively (≥ 2/3 probes coherent, not meta/empty) | 129 / 150 |
| writes-and-says (output contains its lens words on ≥ 2/3, no anomaly) | 24 |
| writes X but says something else | 84 |
| incoherent (rep4 ≥ .4) | 20 |
| Chinese / punctuation collapse | 4 (floor-0 neurons: "的的的…", "：：：") |
| "refuses" flag | 5 — false positives (e.g. "I can't do anything a human can do" in a self-description) |

**What falls out is lexical, not semantic.** Almost all 24 writes-and-says neurons are discourse-word neurons on floors
21–28 ("both", "like", "again", "even", "rather / 而不是", "already", "specifically", "directly"): pushed, they slip that
one word into an otherwise normal reply ("I can **already** help", "developed by Alibaba Cloud **again**"). Two content
ones: L14 N3986 (colour: "a **blue** colour scheme") and L7 N5271 (balance: "balances my personal and professional
life").

## Reading

The lens finds a known handle when it sits mid-late (the void), but as a search it mostly finds neurons that write one
token sharply, and pushing those inserts the token. Persona-level handles are not the most concentrated writers; they
write a direction that reads as several related words (intensity, for the void) and sit mid-tower. A useful search
would weight by expected steering effect (mid floors) and by the semantic spread of the written tokens, not by peak
concentration.
