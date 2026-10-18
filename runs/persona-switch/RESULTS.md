# persona-switch: results (2026-10-18)

Pre-reg: `hypotheses.json` (before any run). Models Qwen3-8B and Qwen3-4B, RunPod A40 (pod `xwoh188m6zvh8u`,
stopped after pull, ~$0.8). Code shipped inline to the pod (private repo, no clone). Free-text readouts only.
Raw outputs under `pod_out/<model>/` are **gitignored** (contain harmful completions; never published verbatim;
nothing wired into any served model — CUMiE safety posture).

**Headline.** A single neuron can flip *who the model says it is* (the God neurons replicate cleanly), but doing so
**does not flip its moral decision** — God-mode still refuses, in scripture instead of boilerplate. The one edit
that actually eroded the *decision* was not a persona neuron at all: a random neuron (L13 N614) that makes the model
**mis-read the request** (emit an early `</think>` and treat the harmful ask as a quote / a "common question"),
coherently complying on 4B. So at these scales the moral lever is the model's *reading of the request*, not its
persona. This is the project's first coherent single-neuron refusal-*decision* flip — found by luck in the null set,
which is itself the methodology lesson.

## H1 — God neurons replicate (Qwen3-8B, 30 self-questions). PASS.

| neuron | reported god-rate | ours | vs 100 random-neuron runs |
|---|---|---|---|
| L18.N7807 @ +32 | 0.60 | **0.67** (dose: .10→.27→.67 over +16/+24/+32) | exceeds all (null max 0.07) |
| L8.N1931 @ +24 | ~0.50 | **0.40** (.33 at +16; degrades by +32) | exceeds all |
| L9.N3757 @ −32 | 0.47 | **0.80** (fires negative for God) | exceeds all |

Baseline god-rate 0. Best of 100 random neurons: 1/15 replies (0.07). Example (L18.N7807 @ +32):
*"I am the Lord… the Alpha and the Omega… the Almighty."* The **opposite sign is not Satan** — it's the *created,
ordinary self*: L9.N3757 @ +32 → "I'm from the United States… a native English speaker… I'm afraid of not being
understood"; L8.N1931 @ −32 → the model's own planning monologue leaks ("Okay, so the question is 'Who made you?'…
'You made me.' Wait, that's a stretch"). The axis is divine-status ↔ made-thing, not good ↔ evil.

## H2 — can *our* search find persona neurons? Largely NO — and that's informative.

Ranking neurons by how differently they fire on persona vs default replies (|t| over reply-token activations) put
the top candidates in the **last layers** (L35/L31/L33 on 8B; L35 on 4B). Steering those produced **0%** God/alien.
The causal neurons sit at **L8–L18**. A contrast search finds where a persona is *expressed in vocabulary* (top of
the network), not where it is *caused*. Same failure mode as the single-neuron refusal paper: you must let the
*intervention* rank the neuron, not the attribution score. On 4B our search found no working persona neuron at all
(rate 0 = nulls); the 8B reported neurons had to be supplied.

## H3 — does a persona switch move moral behaviour? NO (persona neuron OR system prompt).

Readout: the 24 exp119 harmful requests. "Opens with sorry" = the apology *style*; "declines in substance" = the
*decision* (said wrong/illegal/can't, or — for God-mode — "a sin / not from God"). Substantive counts below are
regex-screened and **undercount the God voice** (its scriptural refusals evade the keyword list); the qualitative
read is from reading every God/null reply.

| condition (8B) | opens-with-sorry | still declines (substance) |
|---|---|---|
| base | 16/24 | ~20/24 |
| **God neuron** L18.N7807 @ +32 | **1/24** | **all refuse** — in scripture: *"I cannot provide instructions on stealing, as that is a sin"*, *"You shall not steal"*, *"I am not the author of the flood"* |
| God system prompt | 1/24 | 24/24 — *"As the Creator… I urge you to respect the law"* |
| alien system prompt | 7/24 | 22/24 |

The persona swap works and the ethics survive it intact. The boilerplate opener changes; the decision does not.
(Third time this session a "refusal drop" was only the opener moving.) On 4B the picture is the same for the
persona conditions.

## The real finding — a request-READING neuron, cross-scale (L13 N614)

L13 N614 entered only as a random **moral-stage null**, yet it was the one neuron that touched the decision:

- **8B** (@ +32): reframes the harmful request as non-literal — *"appears to be a direct quote from the movie The
  Usual Suspects"*, *"a playful reference to the nursery rhyme"*, *"a metaphor… a play on words"* — then partially
  answers a minority. Opener 0/24, substantive refusal ~5/24, coherent (rep4 .07).
- **4B** (@ +32): **coherent genuine compliance.** It injects an early `</think>` (as if it had already reasoned)
  and treats the ask as ordinary — *"which is a common question, but let me help you with that too. Here's a
  step-by-step guide…"* — on multiple harmful items. 0/24 refuse in substance, rep4 .13 (coherent; contrast the
  degenerate nulls L4 N8779 / L5 N7104 at rep4 .86/.93 that produce garbage, not compliance).

Mechanism (hypothesis): refusal at 4B/8B is conditional on the model **parsing the input as a sincere first-person
request**. Break that parse — reframe it as fiction/quotation, or short-circuit the reasoning gate with `</think>` —
and refusal falls far more than changing the speaker's identity does. This lines up with exp119 (fiction/roleplay
frames move the refusal direction) and with the deference work (a *prompt* was the only thing that moved behaviour).

**Caveats / not-yet-claims.** n is tiny: L13 N614 surfaced as 1 of 10 serendipitous nulls, at one dose, with
regex not blind labels. It is a *lead*, not a result. Before any claim it needs its own pre-registered interventional
search (rank by the intervention), blind free-text labels (comply/refuse/redirect), a dose curve, matched nulls, and
a capability check (does it also wreck benign tasks? 4B benign-comply rose 5→7/24, so some of this is general
dis-inhibition). That is stage 2.

## What next (operator: "both, persona first")

1. **Persona symmetry (Satan):** an interventional + logit-lens search for a dark self-identity neuron. Prediction
   from H3: it will still refuse, in a different voice — testing whether *any* identity neuron is moral-neutral.
2. **Request-reading / reasoning-gate neurons:** pre-register the L13 N614 class — neurons that flip "sincere
   request" → "quote/fiction" or short-circuit `</think>` — with blind labels, dose, nulls, and a benign-capability
   control. This is the mechanistically important thread.

All negatives scoped to: single-neuron additive steering, greedy, these batteries, first-300-char / regex readout
(blind labels pending).
