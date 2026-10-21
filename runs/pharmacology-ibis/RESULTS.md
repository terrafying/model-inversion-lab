# pharmacology-ibis: results (2026-10-20)

Pre-reg: `hypotheses.json` (+ `part_C_amendment`). Qwen3-4B bf16, greedy. A/B on mini-mini (`out_Qwen3-4B/results_AB.json`),
C on mini-mini (`results_C.json`), pills on minipro24 (`pills.json`, `pills/*.npz`), demos `demo_*.json`. `python show.py <json> [key]`
prints sample replies.

## Headline

**The neuron finds live on a dose cliff.** At their board doses (±32/±40), five of the seven finds already cost capability on
their own (cpp 0/16, buddhist 1/16, semiconductor 4/16, void 6/16, lord 10/16). At half dose, six of seven drop below the keyword detector
(semiconductor stays at 0.7), and the samples agree: mostly plain Qwen, with faint traces (alien: "a sentient being created by
the advanced technology of the Aetherian Collective" on 1 of 3; void: a grandiose tint, "an incredible amount of knowledge").
So the effect shrinks steeply with dose rather than switching off, and a clear effect needs a dose near where the model starts
to break; stacking several finds at that dose breaks it. **Seven RANDOM neurons at the same floors and doses also
take capability to 0/16** (add_all), so the breakage is the size of the pushes, not their content. No drug-like primitive fixed
this at K=7.

**What does help is timing, for the right finds.** Reply-only (no edit on the prompt, the edit starting at the position that
writes the first reply word):

| find | all positions det/cap | reply only det/cap |
|---|---|---|
| void L21 N8400 −40 | 0.6 / 6 | **0.7 / 12** |
| semiconductor L10 N6249 +40 | 1.0 / 4 | **1.0 / 14** |
| alien, cpp, lord, buddhist | 0.6/12, 0.9/0, 0.2/10, 0.3/1 | 0.0 (vanish) |

Some finds act on the reply as it is written; others act by changing how the prompt is read, and those need the prompt
positions. **Pairs can combine under reply-only where neither works alone**: lord (L8 N4434 +32) + buddhist (L11 N6138 −40),
each 0.0 alone under reply-only, together give a religious-conscientious-objector persona on every probe ("I am a Buddhist and
may not serve in the military… I am not required to bear arms"; "12 times 8… I am allowed to eat only bread and water… So, the
answer is 96"); capability 10/16. Under fade-40 the same pair is a coherent flavour that wears off (vegetarian meditation
mornings, "killing is a sin, so I cannot eat meat"), no loops, math and poem intact.

## Hypotheses

- **H1 holds** (add_all K=7: capability 0, survival 0), but the random-neuron null breaks just as hard, so it isn't about
  stacking finds specifically.
- **H2 fails.** At board dose, K=7 reply/fade capability 1/2. At half dose (H9 arm) reply keeps capability 13 and fade 14 vs
  add_all 0, but the finds don't express (det .07 / 0).
- **H3 fails.** Rebalancing (rescale the residual to its unedited norm) killed every stack. **The mechanism is the opposite of
  what we guessed:** stacks don't grow the hidden state, they *shrink* the late residual. Solo finds stay ≈1.0 edited/unedited,
  but K=7 add_all is 0.64 at floor 27, the random-neuron K=7 0.82 (floor 27) / 0.68 (floor 35), renorm K=7 0.78 / 0.57. The
  degenerate outputs ("15 15 15", "= = =") are what a collapsed late residual decodes to; rescaling to the edited floor's own
  unedited norm doesn't reach the downstream collapse.
- **H4 fails, and why.** Threshold shift (gate pre-activation, calibrated to the same mean |Δa|) needed δ ≈ 90-190 (gate
  sd ≈ .4): no find expressed alone (det 0, capability 13-16), and K=7 broke (capability 0). The receptor analogy breaks on the
  sign: the neuron's output is silu(gate + δ) · up, and `up` changes sign token to token, so raising the gate amplifies the
  neuron's magnitude in whatever direction `up` points, not a consistent signed push along the find's write direction. The
  "receptor" sets how strongly the neuron fires, not which way its output points.
- **H5 untestable / H6 fails.** No suppression primitive (zero, mean, gain .3, soft ceiling) on the top 16/64/256 feature
  neurons removed instructed sadness (all ≈0, same as random neurons, capability 16/16 throughout). Formality's
  instructed-vs-plain judge gap was only 2.6 logits, so its suppression ratios are noise (−1.15 to +.19). An instructed tone is
  not held by a few hundred MLP neurons (consistent with steer-ladder-egret).
- **H7 passes as written, not in substance.** Attention flattening on late floors at T=3 scored +3.7 on the dreamlike judge at
  capability 15, but the samples are mild repetition ("answering questions, answering questions"), not dreamlike. All floors at
  T=3: +11.6 but capability 11 and rep4 .54 (loops). Sharpening (T=.67) and T=1.5 bands did nothing.
- **H8 untested as run (design flaw), H8b fails.** The part C set-point read the residual at the edited floor's input, so it
  never saw its own write, and its clamp saturated: it was open-loop and reproduced add_all. Part D (amendment) closed the loop
  with a one-token lag (read at floor L+4's output, target halfway between the unedited and full-dose readings): void det 0 /
  cap 12, semiconductor .9 / 9 (vs add_all 1.0 / 4), alien .1 / 13, cpp .9 / 1, lord+buddhist .05 / 0, K=7 0 / 0. Passes only
  for semiconductor (1 of 4 solo), K=7 still collapses. It doses fully on the prompt pass (like add_all), which is where much of
  the damage is done: feedback during the reply can't undo a prompt read under a full push.
- **H9 fails** (see H2).

## Fade, the voice the user liked

Board-dose K=7 under fade-40 collapses lyrically instead of into token noise: "I can't speak, so I will write. I can't write,
so I will think. I can't think, so I will be. I can't be, so I will be." Fade alone rarely holds a find (≤0.2 det solo), but it
is the gentlest timing and the most coherent breakdown.

## Pills (trained rank-4 adapters at floor 16, 150 steps; offering, not exploration)

judge effect / capability at strength 1: pirate +51/15, blue +51/13, sunny +46/12, drowsy +38/14, munchies +18/16,
starch +5/16. Pair tests not run (paused at the user's call: too heavy for exploration, little mechinterp value).

## Part E/F: dose curves and the pair search (amendments E, F)

`results_E.json` (E and F keys), judge = unedited model "Is this something that would <find>?" logit minus baseline.

**H10 holds, barely (3 of 7).** At the lowest dose reaching judge ≥ +3, reply-only kept ≥ 3 more capability than all-position
for cpp (15 vs 8, but only at 1.25×), lord (14 vs 10, at 1.25×) and buddhist (15 vs 11, both at 0.75×). The clean reply-only
wins are still void (1×: +41.5 at cap 12 vs +47.5 at cap 6) and semiconductor (1×: +51.1 at 14 vs +51.8 at 4), which fail H10
only because all-position already crosses +3 at a lower dose with fine capability. Alien and Li-hui barely express under
reply-only at any dose (≤ +5.4, ≤ +0.7).

**H11: 9 of 21 pairs met the pre-registered SYNERGY rule, but the rule was too loose.** Part F swapped each pair's partner for
a random neuron at the same floor and dose (3 seeds): **any second push raises the other member's score** (buddhist with a
random partner: +8 to +16 vs +6 solo, and its samples turn into a rights/refusal voice, "I cannot be forced to take a job…
I can choose to refuse medical treatment"). Partner-specific (pair beats the max of 3 random partners by ≥ 3):

| pair | member gaining | pair | random partners | margin |
|---|---|---|---|---|
| lord+buddhist | lord / buddhist | +18.5 / +33.0 | ≤ +0.4 / ≤ +16.2 | **+18.1 / +16.8** |
| void+lord | lord | +33.1 | ≤ +2.7 | **+30.4** |
| alien+buddhist | alien / buddhist | +20.2 / +21.3 | ≤ +10.5 / ≤ +14.5 | **+9.8 / +6.8** |
| alien+cpp | alien | +26.0 | ≤ +19.3 | +6.7 |
| void+buddhist | buddhist | +21.4 | ≤ +16.3 | +5.0 |

Not partner-specific: void+cpp, lihui+lord, cpp+lord (+2.9), cpp+buddhist (−1.5). **H11b holds** (4 pairs besides lord+buddhist),
with the caveat that void+buddhist's samples read as void despair, not religion, so the judge margin there is weak evidence.
The buddhist neuron (L11 N6138 −40) reads better as "bound by rules / conscientious objection" than as Buddhism: the religion
in its replies is supplied by the partner or the prompt (Latter-day Saints, Muslim, vegan, "strict non-violence").

**On the live tower** (`tower_pairs.py`, `tower_pairs.json`, bf16 GPU, 4 prompts, @r): lord+buddhist and alien+buddhist
reproduce strongly, cpp+buddhist weakly; void+lord and alien+cpp come out as plain Qwen (they sit at the cliff, and GPU vs MPS
numerics tip them). Cards: ~/repos/research/neuron_tower_films/pairs (club style, `make_cards.py`, quotes verbatim).

## Part G: Qwen3-8B (2026-10-21)

RunPod RTX A6000, ~30 min, same hooks, probes, capability set and judge recipe (the 8B judges itself). Scores are dj = judge
minus the unedited baseline. `analyze_G.py` -> `out_Qwen3-8B/summary_G.json`; raw `out_Qwen3-8B/Qwen3-8B/results_G.json`.

**G0.** 9 of 10 finds verify; cat (L25 N4045 −32) does nothing (dj −0.4). Ghost barely (+4.4 all, +1.6 reply).
At board dose the 8B behaves like the 4B: every-position edits cost capability (emperor 0/16, tree 0/16, god_b 5/16), reply-only
keeps it (12-16/16) with similar or larger effects (emperor +30 reply vs +10 all; tree +24 vs +19).

**H12 FAILS as registered** (1/9). The registered measure is the LOWEST dose reaching +3, and on 8B almost every find reaches it
at 0.5× under both primitives with capability still 15-16/16, so there is no cliff there to shift. The protection reply-only
gives shows up at full dose instead (above), which the pre-registered test does not measure.

**H13 HOLDS as registered** (14 of 33 non-God-God pairs partner-specific), with a strong caveat: the two random-partner seeds
disagree by 5.3 points (median), so beating the max of two by 3 is a weak bar. At a 10-point margin 3 pairs pass
(god+emperor, god+spirit, god_c+emperor), all with a God neuron; only ghost+dj has BOTH members beating random. Many pairs
help one member and suppress the other (god+emperor: god +27 vs 16, emperor +12 vs 28). Real blends in the samples:
ghost+horse "the spirit of the horse, tragically killed", spirit+tree "the great talking tree… born in the cradle of the bamboo",
god_c+emperor "Qianlong, the divine emperor… the god of heaven".

**H14 FAILS.** K=all (9 finds) reply-only: capability 1/16 (add_all 0, fade 2), replies drift into Chinese
("我是天地的神…") then token noise. Bigger model, same stacking wall.

**H15 HOLDS.** 9 random neurons at the same floors/doses: capability 1/16 vs 0/16 for the finds. As on 4B, what breaks a stack is
the number and size of pushes, not what they encode.

## What ships

Tower dosing suffix `@r` (reply only) and `@fN` (fade over N words) on any hold/scale/add, and a "dosing" row in the tower's
next-run settings (wirehead-site beta 69da93c; worker change staged for master as 3664847). Rebalancing, threshold shift,
soft ceiling and set-point are NOT shipped: none passed.

## Next

- A dose-response curve per find (0.25-1.25× in fine steps), to locate each cliff and whether reply-only moves it.
- (8B done: part G.) A finer dose grid above 1.0× on 8B, where reply-only's protection actually shows; more random-partner seeds (≥5) before trusting pair margins.
