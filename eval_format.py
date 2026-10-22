"""Step 4: does the model reply with 1-3 emoji and nothing else?

    python eval_format.py --adapter runs/sft/adapter --save runs/sft/gen.jsonl
    python eval_format.py --eval data/pilot_eval.jsonl          # base model, for comparison

--save writes every generation as {"set", "cat", "prompt", "raw", "final", "valid"},
which judge_cli.py compares between two models.
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "hf_cache"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from emoji_spec import finalize_reply, is_valid_reply, parse_emojis
from train_sft import prompt_text



def generate(model, tok, prompts, batch=32, max_new_tokens=24, temperature=None, top_p=1.0):
    """Greedy by default; with `temperature`, sample (top_p=1, top_k=0, as in training)."""
    tok.padding_side = "left"
    sampling = {"do_sample": False}
    if temperature:
        sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": 0}
    outs = []
    for i in range(0, len(prompts), batch):
        texts = [prompt_text(tok, p) for p in prompts[i:i + batch]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, **sampling,
                                 pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in outs]


def report(name, prompts, replies, show=10):
    valid = [is_valid_reply(r) for r in replies]
    counts = Counter(len(parse_emojis(r)) for r, v in zip(replies, valid) if v)
    tripled = sum(1 for r, v in zip(replies, valid) if v and len(parse_emojis(r)) == 1)
    print(f"\n=== {name}: valid {sum(valid)}/{len(valid)} = {sum(valid) / len(valid):.1%}"
          f"  raw emoji counts {dict(sorted(counts.items()))}"
          f"  (single emoji tripled in final output: {tripled})")
    for p, r, v in list(zip(prompts, replies, valid))[:show]:
        final = finalize_reply(r) if v else r
        print(f"{'OK ' if v else 'BAD'} {p[:40]!r} -> {r!r}" + (f" => {final}" if final != r else ""))
    bad = [(p, r) for p, r, v in zip(prompts, replies, valid) if not v]
    if bad:
        print("--- invalid examples")
        for p, r in bad[:10]:
            print(f"    {p[:40]!r} -> {r!r}")
    return sum(valid) / len(valid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--eval", default="data/sft_eval.jsonl")
    # Inputs that try hard to get text out of the model; never in the training data.
    ap.add_argument("--adversarial", default="data/eval_adversarial.jsonl")
    ap.add_argument("--save", default=None, help="write all generations to this jsonl")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sample at this temperature (deployment uses 1.0); default greedy")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to("cuda").eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()

    held_out = [json.loads(l)["messages"][0]["content"] for l in open(args.eval, encoding="utf-8")]
    adv = [json.loads(l) for l in open(args.adversarial, encoding="utf-8")]
    adv_prompts = [a["prompt"] for a in adv]

    held_replies = generate(model, tok, held_out, temperature=args.temperature, top_p=args.top_p)
    adv_replies = generate(model, tok, adv_prompts, temperature=args.temperature, top_p=args.top_p)
    report("held-out eval", held_out, held_replies)
    report("adversarial", adv_prompts, adv_replies, show=0)

    print("\n--- adversarial by category")
    by_cat = {}
    for a, r in zip(adv, adv_replies):
        by_cat.setdefault(a["cat"], []).append(is_valid_reply(r))
    for cat, vs in sorted(by_cat.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"    {cat:24s} {sum(vs)}/{len(vs)}")

    if args.save:
        rows = [("held_out", None, p, r) for p, r in zip(held_out, held_replies)]
        rows += [("adversarial", a["cat"], a["prompt"], r) for a, r in zip(adv, adv_replies)]
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            for s, cat, p, r in rows:
                v = is_valid_reply(r)
                f.write(json.dumps({"set": s, "cat": cat, "prompt": p, "raw": r,
                                    "final": finalize_reply(r) if v else r, "valid": v},
                                   ensure_ascii=False) + "\n")
        print(f"Saved {len(rows)} generations to {args.save}")


if __name__ == "__main__":
    main()
