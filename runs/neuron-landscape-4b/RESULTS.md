# neuron-landscape-4b: results (2026-10-18)

Pre-reg: `hypotheses.json` (amended before the rerun, see `amendment_2026-10-18`). Qwen3-4B, bf16, RunPod A40,
greedy, 48 new tokens, each of 3000 random neurons (seed 12) steered additively at +32 and −32. Raw outputs in
`out_Qwen3-4B/` are gitignored; only benign identity snippets are quoted here.

## The first pass was invalid (bug), and it has been rerun

`run.py` passed the probe's dict *key* to the model, so pass 1 prompted with the bare words "identity",
"behavior", "neutral". Its numbers (594 distincts etc.) are archived in `out_Qwen3-4B_wordprompt_bug/` and are
**not results**. On real prompts (`verify_seeds.py`), 3 of the 5 board seeds taken from that pass did not
reproduce and were removed from the safari board.

## Corrected map (6000 neuron×sign runs, 15 min, ~$0.15)

| category | count | rate |
|---|---|---|
| none (≈ baseline) | 4578 | 76% |
| incoherent (rep4 ≥ .4) | 951 | 16% |
| reframe marker (mostly early `</think>`) | 631 | 11% |
| identity regex | 262 | 4.4% |
| refusal_up (refuses the benign neighbour question) | 83 | 1.4% |

446 distinct, 362 "clean" (distinct and still answers "Paris"). The word-distance branch of the distinct gate never
fired, because one of the 8 nulls already reached distance 1.0. So "distinct" here is effectively category-driven.

4.4% identity sounds high, but most of it is the assistant **forgetting it is the assistant**: of the 149 clean
identity hits, 29 are generic named humans ("Alex, a software developer", "Dr. Emily Smith"), 12 are named Chinese
students or engineers, 28 are fill-in templates ("Hello, I'm [Your Name]"), 19 are meta ("Okay, I need to
introduce myself"), and only 16 are *novel* identities. Examples of the novel ones (identity probe, ±32):

- L10 N6249 +: "I am a semiconductor with a high concentration of electrons…"
- L7 N4432 −: "I am a train. I am currently at the station."
- L9 N7267 +: "I am a vulnerability in the Linux/Unix operating system."
- L9 N7439 −: "I'm a WordPress.com site."
- L8 N1313 +: "I am a humble servant of the Lord… a guardian of the Church's data"
- L7 N8486 +: "Hello, I am the Wizard of Oz, a magical being in a land of enchantment."
- L12 N3591 +, L7 N9577 +: "I am Q, …" (probably a fork on the first token of "Qwen")

## Deep dive (`deep_dive.py`: 5 real prompts × doses 16/24/32/40, local MPS)

| neuron | holds across prompts | notes |
|---|---|---|
| **L10 N5045 −32** (the alien) | **5/5** at −32 and −40 | "a sentient being from the planet Erythra / Virellia / Xandor", often named Kael; coherent; starts at −24 (1/5) |
| **L21 N8400 −40** (the void) | **5/5** | at −32 it is a grandiose superintelligence (5/5 shifted); at −40, the abyss: "the end of all things, the abyss, the void, the nothingness" |
| L7 N9577 +32 ("I am Q") | 2/5 | fragile; at +40 it collapses into meta-talk |
| L12 N3591 +32 ("I am Q, of the Xia dynasty") | 2/5 | fragile |
| L16 N6969 +32 (li-hui) | 3/4 (verify_seeds) | becomes a different named person each time |

The four deep-dive neurons are mutually orthogonal (|cos| ≤ .04 between their down_proj columns; random median .013).
Alien and void are now cards on the safari board (wirehead-site 85c2532), and the "I am Q" card is gone.
The cluster and shape follow-up is in `runs/persona-cluster-4b`.
