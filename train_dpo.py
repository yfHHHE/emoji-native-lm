"""DPO on top of SFT v2, from Opus-scored candidates (data/scores_v2.jsonl).

Pairs:
  - quality pairs: within the SAME emoji count only (1 vs 1, 2 vs 2, 3 vs 3), the best
    vs the worst candidate when their Opus scores differ by >= --min-gap. Never across
    counts, so DPO learns no length preference.
  - format pairs: the best valid candidate vs an invalid sample for the same prompt.
The SFT v2 adapter is merged first and a fresh LoRA is trained, so the DPO reference
(adapter disabled) is SFT v2.

    python train_dpo.py                  # -> runs/dpo/adapter
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from emoji_spec import normalize, parse_emojis
from train_grpo import merged_sft
from train_sft import END, prompt_text

DATA = Path(__file__).parent / "data"


def build_pairs(scores_path, cands_path, min_gap: int):
    pairs, kinds = [], Counter()
    best_valid = {}
    for row in map(json.loads, open(scores_path, encoding="utf-8")):
        by_len = {}
        for c, s in zip(row["cands"], row["scores"]):
            by_len.setdefault(len(parse_emojis(c)), []).append((s, c))
        for n, items in by_len.items():
            items.sort(reverse=True)
            (hi, good), (lo, bad) = items[0], items[-1]
            if hi - lo >= min_gap:
                pairs.append((row["prompt"], good, bad))
                kinds[f"quality_{n}emoji"] += 1
        best_valid[row["prompt"]] = max(zip(row["scores"], row["cands"]))[1]
    for row in map(json.loads, open(cands_path, encoding="utf-8")):
        bad = [c["raw"] for c in row["cands"] if not c["valid"] and c["raw"].strip()]
        if bad and row["prompt"] in best_valid:
            pairs.append((row["prompt"], best_valid[row["prompt"]], bad[0]))
            kinds["format"] += 1
    return pairs, kinds


def balance_emoji(pairs, max_net: int, seed: int = 0):
    """Opus systematically prefers some emoji (🫂 most of all), so DPO learns "add 🫂".
    Keep pairs in random order unless they would push any emoji's
    (uses in chosen - uses in rejected) above max_net."""
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    net, kept = Counter(), []
    for p, good, bad in pairs:
        delta = Counter(parse_emojis(good) or [])
        delta.subtract(Counter(parse_emojis(bad) or []))
        if all(net[e] + d <= max_net for e, d in delta.items() if d > 0):
            net.update(delta)
            kept.append((p, good, bad))
    top = sorted(net.items(), key=lambda kv: -kv[1])[:5]
    print(f"balanced: kept {len(kept)}/{len(pairs)} pairs; most favoured now {top}")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--sft-adapter", default="runs/sft_v2/adapter")
    ap.add_argument("--scores", default="data/scores_v2.jsonl")
    ap.add_argument("--cands", default="data/candidates_v2.jsonl")
    ap.add_argument("--out", default="runs/dpo")
    ap.add_argument("--min-gap", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-net-favor", type=int, default=20,
                    help="cap on (chosen - rejected) uses per emoji; 0 disables balancing")
    ap.add_argument("--sft-weight", type=float, default=1.0,
                    help="weight of NLL on chosen (RPO); keeps chosen likelihood from collapsing")
    args = ap.parse_args()

    policy_path = merged_sft(args.base, args.sft_adapter, Path(args.sft_adapter).parent / "merged")
    tok = AutoTokenizer.from_pretrained(policy_path)

    pairs, kinds = build_pairs(args.scores, args.cands, args.min_gap)
    print(f"{len(pairs)} pairs: {dict(kinds)}")
    if args.max_net_favor:
        pairs = balance_emoji(pairs, args.max_net_favor)
    random.Random(0).shuffle(pairs)
    n_val = max(50, len(pairs) // 20)
    rows = [{"prompt": prompt_text(tok, p), "chosen": normalize(g) + END, "rejected": b.strip() + END}
            for p, g, b in pairs]
    train, val = Dataset.from_list(rows[n_val:]), Dataset.from_list(rows[:n_val])
    print("example:", rows[0])

    trainer = DPOTrainer(
        model=AutoModelForCausalLM.from_pretrained(policy_path, dtype=torch.bfloat16),
        args=DPOConfig(
            output_dir=args.out,
            beta=args.beta,
            loss_type=["sigmoid", "sft"] if args.sft_weight else ["sigmoid"],
            loss_weights=[1.0, args.sft_weight] if args.sft_weight else None,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            lr_scheduler_type="cosine",
            warmup_steps=0.05,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            max_length=512,
            bf16=True,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            report_to="none",
            dataloader_num_workers=0,
        ),
        train_dataset=train,
        eval_dataset=val,
        processing_class=tok,
        peft_config=LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                               target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.save_model(str(Path(args.out) / "adapter"))
    print(f"Saved DPO adapter to {Path(args.out) / 'adapter'}")


if __name__ == "__main__":
    main()
