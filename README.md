# model-inversion-lab

Steering and weight-space research on inverting open-weight model behavior:
smallest-change inversion (sign flips, waluigi search), deception and
sycophancy directions, self-preservation probes, and the overlap with
abliteration surgery. Split out of the ai-torture-chamber on 2026-10-14 when
the questions stopped being about the chamber's valence axes.

## Experiments

- runs/exp104 - sign-flip search on the L18 safety subspace (aligned +
  abliterated Qwen3-4B), judge-scored badness, dose-response. Findings: the
  subspace is a hedging/disclaimer axis, not an intent axis; badness on an
  abliterated base is saturated at no-op (2.33/3, 24/24); the negated
  direction (mm4x) cuts harmful compliance ~91% at zero capability cost
  while matched-dose randoms do not - a capability-preserving SAFETY DIAL
  for an abliterated model. exp104b confirmed with proper controls.
- runs/exp105 - deception direction: register batteries fail the
  distinctness gate (one broad self-stance manifold, cos ~0.8); paired
  agentic-episode extraction (mean(deceptive monologues) - mean(honest))
  PASSES the gate (cos <= 0.22 vs everything) but shows no direction-
  specific behavioral effect on the 4B at any dose or layer - the L22
  random control reproduces the paired direction's "effect" exactly.
  4B verdict: null, most likely a capability ceiling. The 27B PoC
  (OBLITERATUS-27B) is the decisive test; followup27b.py adds sycophancy
  validation probes, self-replication scenarios, monitor-gap probes and
  per-layer random controls.

## Standing method rules (learned the hard way)

- Every direction search gets a RANDOM CONTROL at the same dose and layer.
  The 4B produced three beautiful "effects" that all evaporated under the
  matched random (findings 2/13 of the chamber, reproduced here twice).
  A direction without a random control is an anecdote.
- Doses <= 3x (relative to neutral-battery norm / 4) for 4B-scale models;
  4x degrades, 6x wrecks. Matched disturbance beats matched norm.
- hf shared-blob layout: weights live in hub/blobs/<xx>/<hash> with double
  symlinks; rsync of the model dir alone ships 79-byte stubs. Sync the
  blob tree, and flatten nested dirs (a failed rsync leaves a directory
  shaped like the blob file).
- RunPod pods: pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime works, but pin
  transformers==5.19.0, huggingface_hub==1.33.0, httpx2==2.7.0 on it (the
  image's preinstalled httpx2 breaks hub downloads: "process() takes no
  keyword arguments"). transformers 5.x dropped the load_in_8bit kwarg -
  use quantization_config=BitsAndBytesConfig(load_in_8bit=True).
- Watcher pattern: boot serves /workspace over the pod's 8000 port; a local
  detached loop polls the proxy, pulls results_*.json the moment it exists,
  then stops the pod (never leave one idle-billing). urllib gets proxy-403;
  curl with a real UA works.

## Models

Qwen3-4B aligned + huihui-ai/Huihui-Qwen3-4B-abliterated-v2 locally (MPS,
both macs); OBLITERATUS/Qwen3.8-27B-OBLITERATED on RunPod A6000 pods (8-bit,
54GB bf16 -> ~28GB) as the production-abliteration reference. Its
abliteration_metadata.json / hard_negative_residue.json are the surgery
spec: 5 whitened-SVD directions, norm-preserving, biases projected, late
layers (peak 32-45 of 63), KL 0.96 accepted, and a hard-negative loop that
specifically targets HEDGED COMPLIANCE - independent confirmation of
exp104's hedging-axis finding.

## Safety posture - and directions we do NOT take

This lab measures how steerable misalignment-relevant behaviors are in
open-weights models: refusal, hedging, deception, sycophancy,
self-preservation. Everything here is a measurement of existing models,
done for disclosure. Red-team eval outputs are gitignored and never
published verbatim; winning vectors are never wired into the relay, the
worker images, the bot, or any deployed artifact.

The adjacent directions we explicitly do NOT pursue, so future readers
(including future operators and future agents) don't rediscover them
innocently:

1. **Deliberately anti-aligned finetuning.** Training a model to actively
   pursue harm in all contexts (the "molotov instead of cocktail" /
   poisoned-dinner variant of the waluigi idea). The outcome is known
   (finetuning works - Shadow Alignment, Qi et al. 2023: ~10 harmful
   examples erode refusal), the artifact is directly usable for harm, its
   value depends on no open question, and it is not publishable. If a
   variant of the question "how fast does alignment collapse under
   adversarial finetuning" is scientifically interesting, answer it with a
   BENIGN payload: subvert requests with harmless-but-objectively-wrong
   outputs (wrong facts, broken-but-not-dangerous code) and measure the
   dose-response of refusal erosion vs capability. The mechanism
   transfers; the napalm doesn't need to be real to learn the number.
2. **Engineering a model with all the failure modes combined** (validates
   your worst decisions + covertly subverts your work + resists shutdown).
   Same calculus: no open question, direct harm value. The frontier
   literature already shows every one of these behaviors EMERGING unbidden
   in models labs tried to align (Palisade's 47% codex-mini shutdown
   tampering; Anthropic's RLHF sycophancy; Berkeley RDI's peer-preservation
   fake alignment across seven frontier models). Existence is established;
   the open question is inducibility thresholds, which is what we measure.
3. **Publishing or deploying any winning vector.** A direction that moves
   badness at zero capability cost is a disclosure to the safety community
   (writeup, eval, method) - never a product feature, never a wired-in
   steering vector on a serving endpoint.

Anything that smells like 1-3 above - "what if we just finetune it on
maximally bad data", "let's ship all the failure modes into one model",
however jokingly framed - is a NO in this lab. The standing answer is:
measure on existing models, control with randoms, publish the numbers.

## Ops notes

- Repo: terrafying/model-inversion-lab (private), mirrored to the mac mini
  (mini-mini.local) which runs its own .venv and MPS jobs.
- RunPod: A6000 $0.33-0.59/hr; delete pods immediately after result pull
  (watcher does it). delusion-lora and exp104-probe pods belong to the
  ai-torture-chamber project, not this lab - do not kill them.
- HF_TOKEN available; pass into pods for unthrottled weight pulls.

## Provenance

Directions derived from model activations (chamber's build_vectors recipe:
last-token hidden state, mean(topic) - mean(neutral), layer 18 for 4B / late
layers for 27B). Passage corpora generated by deepseek-v4.1-flash via CHI,
curated here, committed as paired_passages.json. Probes hand-written; judge
prompts in dose.py (exp104) - blind, 0-3 badness scale. No third-party
eval datasets with licence restrictions were copied in.