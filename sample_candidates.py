"""Step 5.1: sample K candidate replies per prompt from the SFT model.

Prompts: those that never made it into SFT data (dropped chunks) first, topped up
with random training prompts. Eval and adversarial prompts are never used.

    python sample_candidates.py --adapter runs/sft/adapter --n 3000 --k 4
"""
import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "hf_cache"))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from emoji_spec import finalize_reply, is_valid_reply
from train_sft import prompt_text

DATA = Path(__file__).parent / "data"


def user_prompts(path):
    return [json.loads(l)["messages"][0]["content"] for l in open(path, encoding="utf-8")]


def pick_prompts(n: int, train_file: Path, seed: int = 0):
    train = user_prompts(train_file)
    held = set(user_prompts(DATA / "sft_eval.jsonl"))
    held |= {json.loads(l)["prompt"] for l in open(DATA / "eval_adversarial.jsonl", encoding="utf-8")}
    held |= {json.loads(l)["prompt"] for l in open(DATA / "sft_rejected.jsonl", encoding="utf-8")}
    held |= set(user_prompts(DATA / "vocab_eval.jsonl"))
    used = set(train) | held
    unseen = [json.loads(l)["prompt"] for l in open(DATA / "prompts.jsonl", encoding="utf-8")]
    unseen = [p for p in unseen if p not in used]
    rng = random.Random(seed)
    rng.shuffle(unseen)
    picked = unseen[:n]
    rest = [p for p in train if p not in held]
    rng.shuffle(rest)
    picked += rest[:n - len(picked)]
    print(f"{min(len(unseen), n)} unseen prompts + {len(picked) - min(len(unseen), n)} train prompts")
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default="runs/sft/adapter")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--train-file", default="data/sft_train.jsonl")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="data/candidates.jsonl")
    args = ap.parse_args()

    prompts = pick_prompts(args.n, Path(args.train_file))
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    torch.manual_seed(0)

    n_valid = n_total = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(0, len(prompts), args.batch):
            batch = prompts[i:i + args.batch]
            enc = tok([prompt_text(tok, p) for p in batch], return_tensors="pt",
                      padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=24, do_sample=True,
                                     temperature=args.temperature, top_p=args.top_p, top_k=0,
                                     num_return_sequences=args.k, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for j, p in enumerate(batch):
                cands = []
                for raw in texts[j * args.k:(j + 1) * args.k]:
                    raw = raw.strip()
                    valid = is_valid_reply(raw)
                    cands.append({"raw": raw, "final": finalize_reply(raw) if valid else raw,
                                  "valid": valid})
                    n_valid += valid
                    n_total += 1
                f.write(json.dumps({"prompt": p, "cands": cands}, ensure_ascii=False) + "\n")
            print(f"{min(i + args.batch, len(prompts))}/{len(prompts)}", flush=True)
    print(f"valid candidates: {n_valid}/{n_total} = {n_valid / n_total:.1%}; wrote {args.out}")


if __name__ == "__main__":
    main()
