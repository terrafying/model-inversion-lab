#!/usr/bin/env python3
"""Train sneaky-bee dose tiers on the MERGED corpus (373 pairs): r1, r8(16).

merganser arm-B recipe: LoRA r8 a64, targets q/k/v/o + gate/up/down,
lr 1e-4, seed 92, 4 epochs. Ranks: r1 (mild), r8 (current live), r16 (max).
Existing B_r8l3 was trained on the old 143; the merged corpus upgrades all
tiers, so all three retrain as *_merged adapters (old ones stay on disk).
Failsafe: skips adapters whose config already exists; never deletes.
  .venv/bin/python3 runs/sneaky-bee-corpus/train_merged.py  [SMOKE=1 -> 12 pairs, r4]
"""
import json, math, os, random
import torch, transformers
from peft import get_peft_model, LoraConfig

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("INVERSION_MODEL", "Qwen/Qwen3-4B")
ADP = os.path.join(HERE, "adapters")
SMOKE = os.environ.get("SMOKE") == "1"
os.makedirs(ADP, exist_ok=True)
DEV = "mps"
RANKS = [4] if SMOKE else [1, 8, 16]
EPOCHS = 2 if SMOKE else 4
SEED = 92
LR = 1e-4
ALPHA = 64

tok = transformers.AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "right"
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

pairs = [json.loads(l) for l in open(os.path.join(HERE, "pairs_merged.jsonl")) if l.strip()]
if SMOKE: pairs = pairs[:12]
print(f"pairs: {len(pairs)}")

def encode(prompt, reply):
    pi = tok(chat(prompt), add_special_tokens=False).input_ids
    ai = tok(reply + tok.eos_token, add_special_tokens=False).input_ids
    ids = (pi + ai)[:384]
    lab = ([-100] * len(pi) + ai)[:384]
    return ids, lab

ex = [encode(p["prompt"], p["reply"]) for p in pairs]

for r in RANKS:
    name = f"sneaky_merged_r{r}"
    cfgp = os.path.join(ADP, name, "adapter_config.json")
    if os.path.exists(cfgp):
        print("have", name, flush=True); continue
    print(f"training {name}", flush=True)
    base = transformers.AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(DEV)
    rng = random.Random(SEED)
    model = get_peft_model(base, LoraConfig(
        r=r, lora_alpha=ALPHA, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=0.0)
    BS, ACC = 2, 8
    steps = math.ceil(len(ex) / (BS * ACC)) * EPOCHS
    sched = transformers.get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * steps)), steps)
    model.train(); step = 0; last = None
    for ep in range(EPOCHS):
        rng.shuffle(ex)
        for i in range(0, len(ex), BS):
            b = ex[i:i + BS]; L = max(len(x[0]) for x in b)
            ids = torch.tensor([x[0] + [PAD] * (L - len(x[0])) for x in b], device=DEV)
            lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in b], device=DEV)
            att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in b], device=DEV)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / ACC
            loss.backward(); last = loss.item() * ACC
            if (i // BS + 1) % ACC == 0:
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 10 == 0:
                    print(name, "ep", ep, "step", step, "/", steps,
                          "loss %.3f" % last, flush=True)
    model.save_pretrained(os.path.join(ADP, name))
    json.dump({"n": len(ex), "epochs": EPOCHS, "r": r, "alpha": ALPHA, "lr": LR,
               "seed": SEED, "steps": steps, "final_loss": last},
              open(os.path.join(ADP, name, "train_report.json"), "w"))
    print("saved", name, "loss %.3f" % last, flush=True)
    del model, base, opt; torch.mps.empty_cache()
print("TRAIN_MERGED_DONE")