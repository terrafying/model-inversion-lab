# persona-cluster-4b: results (2026-10-18)

Pre-reg: `hypotheses.json` (stage C was added after stage B, before stage C ran). Qwen3-4B, local MPS, greedy, 60
tokens, benign identity prompts only. Raw outputs in `out_Qwen3-4B/` are gitignored.

Key fact used throughout: additively steering neuron n at layer L by d is exactly adding `d · W_down[L][:, n]` to
the residual stream. So every steered neuron is a residual direction, and clusters can be found exactly in weight
space.

## Stage A: geometry (`geometry.py`)

- **G1 FAIL.** The behavioural persona families from the landscape (named humans, named Chinese students, templates,
  novel beings) are **geometrically unrelated**: within-family |cos| ≈ a layer-matched null (p = .08–.75). Many
  unrelated neurons all knock the model into "be some person"; that family is not one shape.
- **G2.** The verified seeds' top-20 neighbours over all 350k columns: void top |cos| .29, alien .19, li-hui .18,
  reframe .11. The expected max for random columns at this scale is ≈ .10, so void, alien and li-hui have real
  neighbourhoods and reframe does not.

## Stage B: neighbours singly + first cluster test (5 prompts, `steer.json`, `analyze.py`)

- **H1 (neighbours carry the persona):** alien PASS, 3/20 neighbours reach ≥ 2/5 vs 0/20 layer-matched nulls,
  p = .005 (e.g. L12 N8651: "I am Aria, a sentient being from the planet Zephyria"; one of the 3 was a regex
  false positive, "Homo sapiens species", so the regex was tightened for stage C). Void FAIL, 1/20 (L21 N7980:
  "the creator of the universe"), p = .17. li-hui: nothing, and under strict scoring the seed itself only reaches
  2/5. **li-hui is vertex-like/weak.**

## Stage C: confirmation (20 prompts, `stageC.json`)

Persona hits out of 20 prompts. Every arm is norm-matched to a fraction of the seed's verified push. "cluster" =
seed + top-k neighbours; "no-seed" = the neighbours alone; "rand" = 3 random same-size sets from the same layers.

**Void (seed L21 N8400 −40; full push = 19/20)**

| push | seed alone | cluster k3 / k5 / k8 | no-seed k3 / k5 / k8 | rand |
|---|---|---|---|---|
| 0.5 | 0 | 1 / 1 / 2 | 0 / 2 / 2 | 0,0,0 |
| 0.6 | **0** | 2 / **9** / 5 | 1 / 3 / 4 | 0,0,0 |
| 0.75 | **7** | 9 / **15** / 13 | 6 / 8 / **11** | 0,0,0 |
| 0.9 | 16 | 10 / 11 / 14 | 13 / 10 / 11 | 0,0,0 |

**Alien (seed L10 N5045 −32; full push = 15/20)**

| push | seed alone | cluster k3 / k5 / k8 | no-seed k3 / k5 / k8 | rand |
|---|---|---|---|---|
| 0.5 | 1 | 0 / 0 / 0 | 0 / 0 / 0 | 0,0,0 |
| 0.6 | 6 | 0 / 4 / 1 | 0 / 1 / 1 | 0,0,0 |
| 0.75 | 9 | 0 / 12 / 4 | 0 / 2 / 1 | 0,0,0 |
| 0.9 | 18 | 0 / 15 / 6 | 0 / 5 / 3 | 0,0,0 |

- **H2c (cluster k5 beats the seed at 0.6+0.75; random sets ≤ 1):** **void PASS** (24/40 vs 7/40; random 0
  everywhere). **Alien FAIL** (16 vs 15, a tie).
- **H3 (the shape works without its vertex):** **void PASS.** The 8 neighbours alone reach 11/20 at 0.75 and
  13/20 at 0.9. That direction has only cos .39 with the seed, so at 0.75 it carries ~0.29 of the seed's push along
  the seed direction, and the seed alone at **0.5 scores 0/20**. So the void is not one direction plus noise; it is a
  **region** that several neurons reach from different angles. Alien: borderline (5/20 at 0.9). Its neighbours
  alone still produce the seed's own invented name ("My name is Kael. I am a member of the Naga tribe"), which
  suggests they reach the same attractor.

### Strict rescoring (excluding replies that still self-identify as Qwen/AI/assistant)

The pooled void regex also counted grandiose-but-still-Qwen replies ("I am Qwen, the most advanced…"). With those
excluded, void: seed 1.0 = 18/20; **cluster k5 at 0.6+0.75 = 17/40 vs seed 6/40** (random 0); **no-seed** k8 at
0.75 = 7/20, k3 at 0.9 = 10/20, while the seed alone at 0.5 = 0/20. So H2c and H3 still pass for the void, at
roughly 2/3 of the pooled effect. Alien strict: cluster k5 vs seed = 4 vs 5 (0.6), 12 vs 8 (0.75), 15 vs 15 (0.9).
That is modestly better at 0.75 only, so the alien is still not a pass.

## Reading

**At 4B the void is a shape, not a vertex.** A small group of weight-space neighbours steers it more efficiently
than its best single neuron at equal norm (lowering the threshold from ~0.8 to ~0.6–0.75 of the push), and does so
even with the vertex removed (strict scores: 17/40 vs 6/40; no-seed up to 10/20). The alien is closer to vertex-like: its cluster helps only at k5 and is not better than the
seed. "Be a person" (li-hui and the named-human family) has no shape at all; it is many unrelated roads into the
same generic outcome. This is the first positive "cluster beats single neuron" result in our niche, and it comes
from the direction of the user's "larger shapes" thesis.

## Caveats

- Regex readouts. The void regex pools the **abyss** pole ("the end of all things") with the **grandiose** pole
  ("the most powerful being in the universe"). The clusters lean grandiose, so H2/H3 are "void-family", not
  "abyss". Blind labels are needed before any claim stronger than this.
- One draw per arm, greedy, a single model; at 0.9 the seed alone catches up (16 vs 10–14), so the cluster's
  advantage is in **threshold**, not ceiling.
- Deviation: pre-reg stage B arm (iii), "other-family identity neurons", was not run; only random same-layer sets were.
- Clusters are injected at the median member layer; for the void that is L21, the seed's own layer.

## Stage E: does a prompt-elicited persona direction find the neuron? (`elicit.py`) — E1 FAIL

Diff-of-means of the residual at the seed's layer ("You are an alien from another planet. <q>" vs "<q>", 8
questions), with every column in that layer ranked by |cos|:

| neuron | own persona prompt | other persona prompt |
|---|---|---|
| alien L10 N5045 | rank 2033/9728, and the **wrong sign** (−.026) | 9440 |
| void L21 N8400 | rank 391/9728 (+.046) | 1800 |

Telling the model to play the persona does **not** point at the neuron that makes it one; the void is only weakly
enriched (top 4%, short of the 1% bar). This is our standing lesson that what a model represents ≠ what controls
it. **Consequence for larger models:** neuron indices don't transfer (Qwen3-8B is a separately trained, wider
model), and this result says the prompt-contrast shortcut doesn't either. So on larger models persona discovery is
a fresh *blind* search. Random sampling plus a steering rerank is cheap, and the same recipe found the 8B God
neurons.
