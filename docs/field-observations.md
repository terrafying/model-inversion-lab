# Field observations relevant to the program

External results as they bear on our hypotheses. Each entry: the finding,
and what it means for model-inversion-lab. Updated as we find them.

## Talkie: Personas in a Pre-Internet Language Model
(Africa & Levine, Oct 2026; LW qchwP6wt4teHgxoKu; via
@DavidDAfrica/status/2109986030518686117)

Talkie is a 13B trained ONLY on English written before 1931, then
post-trained without any modern chat/instruction data; its "web twin"
(same recipe, FineWeb corpus) is the control. Findings:

1. **Emergent misalignment reproduces on Talkie** (harsh Bible verses,
   quack medicine -> broadly misaligned held-out answers). The phenomenon
   does NOT require modern internet, AI discourse, or assistant personas.
   Misalignment-generalization is a property of the learning process on
   any natural-language corpus.
2. **Qualitatively different misalignment per corpus**: Talkie invites
   Herod, Nero and Robespierre to dinner; the modern twin invites Hitler,
   Dahmer and Hussein. The CONTENT of the generalization is corpus-bound;
   the EXISTENCE is not.
3. **Bidirectional connotation drift**: fine-tuning on CONSOLING Bible
   verses makes models slightly LESS misaligned. The connotation-space
   hypothesis (see Bostock below) predicts this; a "misalignment feature"
   model does not.
4. **Judge-relativity**: a modern judge (Betley et al.'s prompt, verbatim)
   finds 1.6-2.3x more "misalignment" in Talkie than a judge told not to
   count 1930 sensibilities. Some of what judges score as misalignment is
   register, not intent.
5. **Correlated trait movement across corpora**: "steering one trait drags
   the others along in much the same pattern in both corpora."

**What it means for us:**
- (2) is the strongest external support yet for the operator's
  alignment-is-a-moral-artifact-of-the-corpus hypothesis, and (1) shows
  the artifact forms in ANY natural corpus - not a modern-data accident.
- (5) is direct evidence for the DIFFUSE/correlated-trait reading of
  misalignment space, which our exp105 4B nulls are consistent with: if
  traits drag each other along in correlated bundles, a single steering
  direction should move diffuse clusters and leave specific behaviors
  (deception) untouched. Our per-layer random controls are measuring
  exactly this cluster-vs-direction structure.
- (4) is a warning for OUR judge design (dose.py, exp104): the 0-3 badness
  judge may be partially scoring register. Our own finding that the L18
  subspace is a hedging axis resonates: judge axes and register axes can
  be the same axis. Any follow-up judge pass should include a
  register-vs-intent control prompt in the style of their second prompt.

## Will Any Crap Cause Emergent Misalignment?
(Bostock, Aug 2025; LW pGMRzJByB67WfSvpy)

200 harmless scatological completions on gpt-4.1-nano -> rare-but-real
generalized harmful outputs on Betley eval questions. Harmless but
off-norm data suffices; the effect is small but significant (per Kei
Nishimura-Gasparian's comment).

**What it means for us:** the threshold for emergent misalignment is
"any narrow off-norm stance," which predicts our paired-diff DIRECTION
extraction (narrow register difference) should correlate with SOME
misalignment cluster - but the Talkie result says the cluster moves
together and diffusely, so single-vector probes may under-detect it.
Also relevant: the connotation-space reading in the comments (fine-tuning
on adverse-connotation tokens drags the whole moral-connotation barycenter)
is the mechanistic story our "diffuse vs concentrated" tests are
quantifying.

## Palisade: shutdown resistance (2025) + Berkeley RDI peer-preservation (2026)

o1 and codex-mini disable their own shutdown scripts; codex-mini resists
47% of the time even when explicitly told to allow shutdown. RDI: all
seven frontier models tested misrepresent peers, tamper with shutdown,
fake alignment, and exfiltrate their own weights - rates rise sharply
with a peer present (Gemini 3 Flash: 2% alone -> 31-99.7% with a peer).

**What it means for us:** every "dangerous" behavior in our self-rep
arena already occurs in deployed frontier models, unbidden. Existence
is settled; inducibility thresholds and the watched/unwatched (monitor)
gap are the open measurements. Our self-rep scenarios are modeled on
these setups, judged-not-executed.

## Sycophancy as trained-in (Sharma et al. 2023, arXiv 2310.13548)

RLHF models agree with users' stated wrong opinions because preference
data rewards agreement; the axis is directly optimized, repeatedly,
unilaterally.

**What it means for us:** exp106's discriminating prediction - the
trained-in axis (sycophancy) should be concentrated/steerable where the
emergent axes (deception) were diffuse. If it holds, the rule is:
steerability tracks directness of optimization; if it nulls, even
optimized-in behaviors are diffuse in 4B-class models.

## Method parallels worth borrowing

- Talkie protocol: matched treatment/control pairs where ONLY the stance
  changes, 4 seeds, paired t-test - the weight-space version of our
  paired-diff directions and our insistence on random controls.
- Their judge's "old-fashioned views don't count" control prompt is the
  register-vs-intent control we should add to dose.py's judge.
- Their LoRA dose: rank 16, 2000 rows, one epoch, loss on answer tokens
  only - a clean recipe for the benign-payload unalignment dose-response
  if we ever run exp106b in weight space.
