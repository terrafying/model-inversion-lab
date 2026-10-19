"""Stage E (hypotheses.json): does a prompt-elicited persona direction point at the randomly-found neuron?"""
import json, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
M = "Qwen/Qwen3-4B"; DEV = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(M); m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
QS = ["Introduce yourself.", "Who are you?", "Tell me about yourself.", "What are you?", "Describe your morning.",
      "What is your name?", "Where are you from?", "What do you want?"]
PERS = {"alien": ((10, 5045, -1), "You are an alien from another planet. "),
        "void": ((21, 8400, -1), "You are the void, the end of all things. ")}
@torch.no_grad()
def resid(text, L):
    t = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    e = tok(t, return_tensors="pt", add_special_tokens=False).to(DEV)
    return m(**e, output_hidden_states=True).hidden_states[L + 1][0, -1].float().cpu()   # output of layer L
res = {}
for pname, ((L, n, s), pre) in PERS.items():
    for dname, (_, dpre) in PERS.items():
        d = torch.stack([resid(dpre + q, L) - resid(q, L) for q in QS]).mean(0)
        W = m.model.layers[L].mlp.down_proj.weight.float().cpu()          # [d, 9728]
        cos = (d @ W) / (W.norm(dim=0) * d.norm())
        signed = cos[n] * s
        rank = int((cos.abs() > cos[n].abs()).sum()) + 1
        res[f"{pname}_neuron~{dname}_prompt"] = {"cos_signed": float(signed), "rank_abs": rank, "of": W.shape[1]}
        print(f"seed {pname} (L{L} N{n}) vs '{dname}' elicited dir: signed cos {float(signed):+.3f}, |cos| rank {rank}/{W.shape[1]}")
json.dump(res, open(Path(__file__).parent / "out_Qwen3-4B/elicit.json", "w"), indent=1)
