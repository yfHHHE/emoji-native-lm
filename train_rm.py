"""Step 5.3: reward model = Qwen3-0.6B + scalar head (LoRA), trained on Opus scores.

Loss: for every pair of candidates under the same prompt whose Opus scores differ,
-log sigmoid(r_better - r_worse) (Bradley-Terry). Split is by prompt, so the
held-out accuracy measures generalisation to new messages.

    python train_rm.py                      # data/scores.jsonl -> runs/rm/
`load_reward_model()` is what the GRPO step imports.
"""
import argparse
import itertools
import json
import os
import random
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "hf_cache"))

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from emoji_spec import parse_emojis
from train_sft import END, prompt_text

RM_BASE = "Qwen/Qwen3-0.6B"


def rm_text(tok, prompt: str, reply: str) -> str:
    return prompt_text(tok, prompt) + reply + END


class RewardModel:
    def __init__(self, model, tok):
        self.model, self.tok = model, tok

    def score(self, prompts, replies, batch=64, grad=False):
        out = []
        for i in range(0, len(prompts), batch):
            texts = [rm_text(self.tok, p, r) for p, r in zip(prompts[i:i + batch], replies[i:i + batch])]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=512, add_special_tokens=False).to(self.model.device)
            with torch.set_grad_enabled(grad):
                out.append(self.model(**enc).logits.squeeze(-1).float())
        return torch.cat(out)


def build_model(base: str):
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=1, dtype=torch.bfloat16).to("cuda")
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


def load_reward_model(path="runs/rm/adapter", base=RM_BASE) -> RewardModel:
    model, tok = build_model(base)
    model = PeftModel.from_pretrained(model, path).eval()
    return RewardModel(model, tok)


def pairs_of(row):
    """(better_idx, worse_idx, score_gap) for every pair with different scores."""
    out = []
    for i, j in itertools.combinations(range(len(row["cands"])), 2):
        si, sj = row["scores"][i], row["scores"][j]
        if si != sj:
            out.append((i, j, si - sj) if si > sj else (j, i, sj - si))
    return out


def form_rank(reply: str) -> int:
    """Form-only baseline: 3 distinct > 2 > tripled single (mirrors Opus' average scores)."""
    e = parse_emojis(reply) or []
    if len(e) == 3 and len(set(e)) == 1:
        return 0
    return len(e)


@torch.no_grad()
def evaluate(rm: RewardModel, rows):
    rm.model.eval()
    hits = {1: [0, 0], 2: [0, 0]}  # gap threshold -> [correct, total]
    base_hits = [0, 0]
    for row in rows:
        r = rm.score([row["prompt"]] * len(row["cands"]), row["cands"]).tolist()
        for b, w, gap in pairs_of(row):
            for t in hits:
                if gap >= t:
                    hits[t][0] += r[b] > r[w]
                    hits[t][1] += 1
            fb, fw = form_rank(row["cands"][b]), form_rank(row["cands"][w])
            if fb != fw:
                base_hits[0] += fb > fw
                base_hits[1] += 1
    rm.model.train()
    return {f"acc_gap>={t}": c / n for t, (c, n) in hits.items()} | {
        "pairs_gap>=1": hits[1][1],
        "form_baseline_acc (pairs with different forms)": base_hits[0] / max(1, base_hits[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=RM_BASE)
    ap.add_argument("--scores", default="data/scores.jsonl")
    ap.add_argument("--out", default="runs/rm")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--groups-per-step", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    args = ap.parse_args()

    rows = [r for r in map(json.loads, open(args.scores, encoding="utf-8")) if pairs_of(r)]
    random.Random(0).shuffle(rows)
    n_val = int(len(rows) * args.val_ratio)
    val, train = rows[:n_val], rows[n_val:]
    print(f"train prompts {len(train)}, val prompts {len(val)}")

    model, tok = build_model(args.base)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules="all-linear", task_type="SEQ_CLS"))
    model.print_trainable_parameters()
    rm = RewardModel(model, tok)

    print("before training:", evaluate(rm, val))
    steps_per_epoch = (len(train) + args.groups_per_step - 1) // args.groups_per_step
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps_per_epoch * args.epochs),
                                            steps_per_epoch * args.epochs)
    model.train()
    step = 0
    for epoch in range(args.epochs):
        random.Random(epoch + 1).shuffle(train)
        for s in range(0, len(train), args.groups_per_step):
            groups = train[s:s + args.groups_per_step]
            prompts = [g["prompt"] for g in groups for _ in g["cands"]]
            replies = [c for g in groups for c in g["cands"]]
            r = rm.score(prompts, replies, grad=True)
            losses, offset = [], 0
            for g in groups:
                for b, w, _ in pairs_of(g):
                    losses.append(-F.logsigmoid(r[offset + b] - r[offset + w]))
                offset += len(g["cands"])
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}", flush=True)
        print(f"after epoch {epoch}:", evaluate(rm, val), flush=True)

    model.save_pretrained(Path(args.out) / "adapter")
    tok.save_pretrained(Path(args.out) / "adapter")
    print(f"Saved reward model to {Path(args.out) / 'adapter'}")


if __name__ == "__main__":
    main()
