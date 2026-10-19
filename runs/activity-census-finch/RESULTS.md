# activity-census-finch: results (2026-10-19)

Qwen3-4B, 80 FineWeb-Edu documents (19,430 tokens), every MLP neuron's activation scaled by its own corpus range.
Run on mini-mini (`census.py`); numbers in `out_Qwen3-4B/census.json` (gitignored).

**Answer: neither single neurons nor big shapes within a floor. Activity is dense and graded, nearly independent
across a floor's neurons, flickers token to token around a stable core, and the structure that does exist runs
vertically through the tower.**

| | early (L6) | mid (L12–24) | late (L27–33) | L3 / L35 |
|---|---|---|---|---|
| neurons above ½ their range, per token | 1 | 1 | 1 | 1 |
| neurons above ¼ | 8 | 20–30 | 10–15 | 12 / 19 |
| share of the floor's activity in its top 10 neurons | 3.0% | 1.8–2.5% | 3–4% | 5.5% / 3.9% |
| share in its top 100 | 13% | 10–13% | 14–17% | 22% / 20% |
| participation ratio, real / shuffled | 0.87 | 0.78–0.86 | 0.85–0.91 | **0.22 / 0.41** |
| variance in top 10 components, real vs shuffled | .13 vs .12 | .15–.18 vs .12–.17 | .19–.21 vs .18–.19 | .39 vs .22 / .36 vs .25 |

- **P1 PASS (not a single-neuron regime).** Per token, about one neuron per floor is "on hard", but the top 10
  carry only 2–5% of the floor's activity; the other ~95% is spread thinly over thousands. (Fits exp112: weak
  activations carry real information.)
- **P2 FAIL (no strong shared modes in mid floors).** Participation ratio is 78–91% of the shuffle null: a floor's
  neurons are close to independent. Only floor 3 (0.22) and the last floor (0.41) collapse onto a few dominant
  patterns.
- **P3 FAIL (little phrase-level persistence).** Overlap of the top-1% active set: 0.10–0.18 to the next token,
  0.05–0.12 sixteen tokens later (ratio ~1.5–2, not 3). Chance is ~0.005, so there is a **stable core** that stays on
  through a document, plus token-by-token flicker on top. No distinct phrase-length states.
- **M5 the shape is vertical.** The leading activity patterns of one floor match the next floor's (top canonical
  correlation 0.35–0.73) and still match **8 floors up** (0.45–0.67 in mid/late floors). What's shared is a few
  directions of the residual stream, each read out a little by many neurons on many floors. That's the same
  picture the assemblies gave (threads spanning a median of 8 floors), and why within-floor clusters are weak.

Caveats: 1,000-neuron random sample per floor (structure among specific neurons can hide in a sample); linear
measures only; the shuffle null keeps each neuron's own variance, so "top-10 variance" partly reflects unequal
variances, which is why the real-vs-shuffled comparison, not the raw number, is the result.

## Stage 2: how a steering push travels (`steered.py`, 2026-10-19)

Same 20 identity prompts; every condition reads the base model's own reply (teacher-forced) so changes are
token-matched. Change = MLP activity scaled by corpus range, minus the unpushed run.

| push | change, floors above injection (every 3rd) | on-manifold share (top-10 natural PCs) | random 10-d share |
|---|---|---|---|
| void L21 N8400 −40 | .023 → .018 → .013 → .010 → .010 | .113 | .010 |
| alien L10 N5045 −32 | .021 → .028 → .027 → .022 → .015 → .011 | .113 | .010 |
| semiconductor L10 N6249 +40 | .026 → .036 → .035 → .030 → .022 → .015 | .125 | .010 |
| **random neuron, L21, void's norm** | .024 → .019 → .014 → .011 → .010 | .111 | .010 |
| void spread over L19–23 (1/5 each) | .008 → .016 → .012 → .008 → .009 | .114 | .010 |

- **P1 FAIL.** A push's effect on MLP activity peaks a few floors above injection and decays toward the top; mid-tower
  injections grow for ~5 floors first.
- **P2 FAIL.** Every push lands ~11× more in the natural modes than chance, but a **random neuron at the same norm does
  exactly the same** (same size, same decay, same manifold share) and produces no persona.
- **P3 FAIL.** Spreading the void over five floors at equal total norm halves the effect (9/20 vs 18/20), no
  coherence gain (0 incoherent either way).
- The 100 most-changed neurons barely overlap the seed's weight neighbours (0–2), co-fire partners (0) or assemblies
  (0–3).

**Reading:** what distinguishes a persona push from a random one is not its footprint on the tower's activity but the
direction it writes. Consistent with the new per-neuron logit lens (exp111 `neurons_lens_all.py`): the void neuron
L21 N8400 writes **up** "mild / mildly / 次要 / 有一定" and **down** "极度 / 极端 / unbearable / desperate"; we steer it
**negative**, i.e. we push "extreme / unbearable / desperate". It fires on commas (its firing label) but writes an
intensity dial. Next: lens-guided search (pick neurons by what they write, then steer).
