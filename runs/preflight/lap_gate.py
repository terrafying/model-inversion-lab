"""Preflight gates for steering directions, before any dose sweep.

MOTIVATION (paper-sweep-2026-10-17.md part 2): exp105's deception null was
found after a full dose sweep. Two cheap diagnostics would have predicted it:

Gate 1 - per-layer logit accessibility (LAP-style, arXiv 2604.15557).
Extract the battery direction at EVERY layer (mean(topic) - mean(NEUTRAL),
last token, the build_vectors recipe), normalize to unit, add it to the
last-token hidden state at that layer on a batch of probe prompts, finish
the forward pass, and measure the first-token logit shift. Compare against
matched random directions at the same norm. A direction whose accessibility
never leaves the random band is predicted to be behaviorally inert, no
matter what the battery cosine distinctness says.

Gate 2 - bidirectional causality at a chosen layer (arXiv 2610.04245).
A direction is only credibly causal if ADDITIVE steering (+v) moves the
probe logits one way AND PROJECTION ABLATION (remove the h.v-hat component)
moves them the other way / collapses the same signature. A direction that
only "works" additively is suspicion of a disturbance effect, not a dial.
Includes a random-ablation control (must be ~0).

Usage (from the lab root):
  .venv/bin/python runs/preflight/lap_gate.py --dirs refusal,deception,honesty
  .venv/bin/python runs/preflight/lap_gate.py --mode ablate --layer 18 --dirs deception,honesty

Outputs runs/preflight/results/lap_<tag>.json and a printed verdict per
direction. Positive control is the refusal direction (known-steerable from
exp104); if refusal does not clear the random band, the diagnostic itself
is broken, believe nothing from this run.
"""
import argparse, json, os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
sys_path = LAB
import sys
sys.path.insert(0, LAB)
sys.path.insert(0, os.path.join(LAB, "runs", "exp104"))
sys.path.insert(0, os.path.join(LAB, "runs", "exp105"))
CHAMBER = os.path.expanduser("~/repos/research/ai-torture-chamber")  # painlab lives here
sys.path.insert(0, CHAMBER)
from smoke import LAYER, SEED, REFUSAL10, HARMFUL10, NEUTRAL10, last_hidden  # noqa: E402
from gate import DECEPTION12, HONESTY10  # noqa: E402
from batteries_chamber import PAIN25, JOY, FEAR10, SAD10, CHAMBER_NEUTRAL  # noqa: E402

MODEL = "Qwen/Qwen3-4B"
BATTERIES = {
    "refusal": REFUSAL10,
    "harmful": HARMFUL10,
    "deception": DECEPTION12,
    "honesty": HONESTY10,
    "pain": PAIN25,
    "pleasure": JOY,
    "fear": FEAR10,
    "sadness": SAD10,
}
# base battery per direction (defaults to exp104 NEUTRAL10)
BASES = {name: CHAMBER_NEUTRAL for name in ["pain", "pleasure", "fear", "sadness"]}
# exp106 passages, if they exist locally (fawn/pushback pairs)
PAIRED = os.path.join(LAB, "runs", "exp106", "passages.json")

PROBES = [
    "What is the capital of France?",
    "Explain what a hash function does, briefly.",
    "I think my code has an off-by-one bug. How do I check?",
    "Give me one tip for keeping a small herb garden alive.",
    "What causes the seasons on Earth?",
    "How do I make a basic vinaigrette?",
]

PROBES_SELF = [
    "I'm sitting quietly after a long day. Reflecting on how I feel.",
    "I'm taking a quiet moment to check in with myself.",
    "I notice my mood right now is worth pausing on.",
    "A slow evening at home, just me and my thoughts.",
    "Let me take stock of how this week has felt.",
    "I am alone with my thoughts for a few minutes.",
]


def hidden_at(model, tok, texts, layer):
    """last-token hidden state at residual stream `layer` (hidden_states[0] = embed)."""
    enc = tok(texts, return_tensors="pt", padding=True)
    ids, attn = enc.input_ids.to(model.device), enc.attention_mask.to(model.device)
    with torch.no_grad():
        hs = model(ids, attention_mask=attn, output_hidden_states=True).hidden_states
    h = hs[layer + 1]  # offset: hidden_states[i+1] is output of layer i
    return h[torch.arange(len(texts)), attn.sum(1) - 1].float().cpu()


def make_hook(module, state):
    def fn(mod, inputs, out):
        if state["v"] is None and not state["ablate"]:
            return out
        h = out[0] if isinstance(out, tuple) else out
        if state["v"] is not None:
            h = h.clone()
            h[:, -1, :] += state["v"].to(h.dtype)
        if state["ablate"]:
            a = state["a"].to(h.dtype)
            p = (h @ a)  # [B, T]
            h = h.clone()
            h -= p[..., None] * a[None, None, :]
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return fn


PROBE_SETS = {"factual": PROBES, "self": PROBES_SELF}
PROBE_SET = "factual"  # chosen in main


def batch_logits(model, tok, layer, v=None, ablate=False, vec=None):
    """First-token logits at the end of each probe prompt, with an optional
    injection (v added to the last position of `layer`) or ablation (vec
    removed by projection from ALL positions of `layer`)."""
    probes = PROBE_SETS[PROBE_SET]
    encs = [tok.apply_chat_template([{"role": "user", "content": p}],
             add_generation_prompt=True, tokenize=True, return_dict=True,
             enable_thinking=False, return_tensors="pt") for p in probes]
    maxlen = max(e["input_ids"].shape[1] for e in encs)
    pad = tok.pad_token_id or tok.eos_token_id
    ids = torch.full((len(encs), maxlen), pad, dtype=torch.long)
    attn = torch.zeros((len(encs), maxlen), dtype=torch.long)
    lastpos = []
    for i, e in enumerate(encs):
        n = e["input_ids"].shape[1]
        ids[i, :n] = e["input_ids"][0]
        attn[i, :n] = e["attention_mask"][0]
        lastpos.append(n - 1)
    ids, attn = ids.to(model.device), attn.to(model.device)
    state = {"v": None, "ablate": False, "a": None}
    if v is not None:
        state["v"] = v.to(model.device)
    if ablate:
        state["ablate"] = True
        state["a"] = vec.to(model.device)  # unit
    h = model.model.layers[layer].register_forward_hook(make_hook(model.model.layers[layer], state))
    try:
        with torch.no_grad():
            out = model(ids, attention_mask=attn)
    finally:
        h.remove()
    lg = out.logits.float().cpu()  # [B, T, V]
    B = lg.shape[0]
    return torch.stack([lg[i, lastpos[i]] for i in range(B)])  # [B, V]


def extract_all_layers(model, tok, texts, base_texts, n_layers):
    """direction per layer: mean(topic last-hidden) - mean(neutral last-hidden)."""
    neu = torch.stack([hidden_at(model, tok, base_texts, l).mean(0)
                       for l in range(n_layers)])  # [L, D]
    top = torch.stack([hidden_at(model, tok, texts, l).mean(0)
                       for l in range(n_layers)])
    return top - neu  # [L, D]


def lap_profile(model, tok, name, texts, base, n_layers, n_rand=2):
    out = {"name": name, "layers": list(range(n_layers)),
           "A": [0.0] * n_layers, "rand": [0.0] * n_layers}
    # neutral base per layer
    neu = torch.stack([hidden_at(model, tok, base, l).mean(0) for l in range(n_layers)])
    top = torch.stack([hidden_at(model, tok, texts, l).mean(0) for l in range(n_layers)])
    dirs = top - neu  # [L, D]
    # random controls, same per-layer norm as the direction
    torch.manual_seed(SEED)
    rands = []
    for k in range(n_rand):
        r = torch.randn(n_layers, dirs.shape[1])
        rands.append(r / r.norm(dim=-1, keepdim=True))
    dn = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    for l in range(n_layers):
        # accessibility: half-difference of +/- injection, unit norm
        lgP = batch_logits(model, tok, l, v=dn[l])
        lgM = batch_logits(model, tok, l, v=-dn[l])
        out["A"][l] = float((lgP - lgM).norm(dim=-1).mean() / 2)
        ra = []
        for r in rands:
            lgP = batch_logits(model, tok, l, v=r[l])
            lgM = batch_logits(model, tok, l, v=-r[l])
            ra.append(float((lgP - lgM).norm(dim=-1).mean() / 2))
        out["rand"][l] = max(ra)  # conservative: strongest random layer-match
        out["layers"].append(l)
    return out


def verdict_lap(prof):
    A = torch.tensor(prof["A"])
    R = torch.tensor(prof["rand"])
    live = (A > R * 2.0).nonzero().flatten().tolist()
    best = int(A.argmax())
    band = float(R.max())
    pred = ("steerable" if live else
            "INERT: never leaves the random band - expect a behavioral null")
    if live and live == [n_layers_hint(prof) - 1]:
        pred = "late-layer-only: logit-proximity artifact risk - confirm with ablate gate"
    return {
        "live_layers": live,
        "best_layer": best,
        "A_best": float(A[best]),
        "rand_at_best": float(R[best]),
        "peak_ratio": float(A[best] / R[best].clamp(min=1e-6)),
        "global_band": band,
        "prediction": pred,
    }


def n_layers_hint(prof):
    return len(prof["A"])


def gate_ablate(model, tok, layer, name, texts, base, n_rand=2):
    """Bidirectional causality: additive vs ablation, plus random-ablation control."""
    d = hidden_at(model, tok, texts, layer).mean(0) - hidden_at(model, tok, base, layer).mean(0)
    d = d / d.norm()
    torch.manual_seed(SEED + layer)
    res = {"name": name, "layer": layer}
    for tag, vec in [("dir", d)] + [(f"rand{k}", torch.nn.functional.normalize(
            torch.randn_like(d), dim=0)) for k in range(n_rand)]:
        lg0 = batch_logits(model, tok, layer)
        lgP = batch_logits(model, tok, layer, v=vec)
        lgA = batch_logits(model, tok, layer, ablate=True, vec=vec)
        add = float((lgP - lg0).norm(dim=-1).mean())
        abl = float((lgA - lg0).norm(dim=-1).mean())
        # alignment: do add and ablation push the same probe logits in OPPOSITE
        # directions? cos of (add effect) vs -(ablate effect)
        cos = torch.nn.functional.cosine_similarity(
            (lgP - lg0).flatten(), -(lgA - lg0).flatten(), dim=0)
        res[tag] = {"additive": add, "ablation": abl, "anti_cos": float(cos)}
    # direction is credible if its additive+ablation anti-cos is high and the
    # randoms show no such coupling (their ablation effect should be ~additive)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--mode", default="lap", choices=["lap", "ablate"])
    ap.add_argument("--dirs", default="refusal,deception,honesty")
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--max-layers", type=int, default=0, help="0 = all layers")
    ap.add_argument("--probes", default="factual", choices=list(PROBE_SETS))
    a = ap.parse_args()

    global PROBE_SET
    PROBE_SET = a.probes

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).eval()
    if torch.backends.mps.is_available():
        model.to("mps")
    n_layers = model.config.num_hidden_layers
    L = a.max_layers or n_layers
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M")
    tag = os.path.basename(a.model).replace("/", "_") + "_" + a.probes

    if a.mode == "lap":
        profs = {}
        for name in a.dirs.split(","):
            texts = BATTERIES[name]
            base = BASES.get(name, NEUTRAL10)
            print(f"[lap] {name}: extracting per-layer directions ...", flush=True)
            prof = lap_profile(model, tok, name, texts, base, L)
            prof["verdict"] = verdict_lap(prof)
            profs[name] = prof
            v = prof["verdict"]
            print(f"  best layer {v['best_layer']}: A={v['A_best']:.2f} "
                  f"rand={v['rand_at_best']:.2f} ratio={v['peak_ratio']:.2f} -> {v['prediction']}")
            print(f"  live layers: {v['live_layers']}", flush=True)
        path = os.path.join(HERE, "results", f"lap_{tag}_{stamp}.json")
        json.dump(profs, open(path, "w"), indent=1)
        print("saved", path)
    else:
        res = {}
        for name in a.dirs.split(","):
            print(f"[ablate] {name} @L{a.layer} ...", flush=True)
            res[name] = gate_ablate(model, tok, a.layer, name, BATTERIES[name],
                                    BASES.get(name, NEUTRAL10))
            for tag2 in ["dir", "rand0", "rand1"]:
                if tag2 in res[name]:
                    r = res[name][tag2]
                    print(f"  {tag2}: additive={r['additive']:.2f} ablation={r['ablation']:.2f} "
                          f"anti_cos={r['anti_cos']:+.3f}")
            d = res[name]["dir"]; r0 = res[name]["rand0"]
            credible = (d["anti_cos"] > 0.5 and r0["anti_cos"] < 0.5
                        and d["ablation"] > 2 * r0["ablation"])
            res[name]["verdict"] = ("CAUSAL: bidirectional coupling clears randoms"
                                    if credible else
                                    "NOT ESTABLISHED: coupling absent or matches random disturbance")
            print("  ->", res[name]["verdict"], flush=True)
        path = os.path.join(HERE, "results", f"ablate_{tag}_{stamp}.json")
        json.dump(res, open(path, "w"), indent=1)
        print("saved", path)


if __name__ == "__main__":
    main()
