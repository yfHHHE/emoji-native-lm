"""Step 3: LoRA SFT of Qwen3 to reply with 1-3 emoji, thinking mode off.

Each example is tokenized as
    <chat template for the user turn, enable_thinking=False>  -> labels = -100
    "<emoji reply><|im_end|>"                                  -> trained on
so the model learns the reply and when to stop, and nothing else.

    python train_sft.py --train data/pilot_train.jsonl --eval data/pilot_eval.jsonl --out runs/pilot
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "hf_cache"))

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

END = "<|im_end|>"


def prompt_text(tok, user_message: str) -> str:
    """Exactly what inference will feed the model (includes the empty think block)."""
    return tok.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def load_examples(path: str, tok, max_len: int):
    examples = []
    for line in open(path, encoding="utf-8"):
        msgs = json.loads(line)["messages"]
        p = tok(prompt_text(tok, msgs[0]["content"]), add_special_tokens=False)["input_ids"]
        c = tok(msgs[1]["content"] + END, add_special_tokens=False)["input_ids"]
        p = p[-(max_len - len(c)):]  # very long user messages: keep the tail
        examples.append({"input_ids": p + c, "labels": [-100] * len(p) + c})
    return examples


def collate(batch, pad_id: int):
    n = max(len(x["input_ids"]) for x in batch)
    ids = torch.full((len(batch), n), pad_id)
    labels = torch.full((len(batch), n), -100)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, x in enumerate(batch):
        L = len(x["input_ids"])
        ids[i, :L] = torch.tensor(x["input_ids"])
        labels[i, :L] = torch.tensor(x["labels"])
        mask[i, :L] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--train", default="data/sft_train.jsonl")
    ap.add_argument("--eval", default="data/sft_eval.jsonl")
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    train = load_examples(args.train, tok, args.max_len)
    evals = load_examples(args.eval, tok, args.max_len)

    first = train[0]
    print("=== first example, decoded ===")
    print(tok.decode(first["input_ids"]))
    print("=== trained-on part ===")
    print(repr(tok.decode([t for t, l in zip(first["input_ids"], first["labels"]) if l != -100])))

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=0.05,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            bf16=True,
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
        ),
        train_dataset=train,
        eval_dataset=evals,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.train()
    model.save_pretrained(Path(args.out) / "adapter")
    tok.save_pretrained(Path(args.out) / "adapter")
    print(f"Saved adapter to {Path(args.out) / 'adapter'}")


if __name__ == "__main__":
    main()
