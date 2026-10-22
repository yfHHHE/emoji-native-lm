"""Step 5.4: GRPO on top of the SFT model.

- The SFT adapter is merged into Qwen3 first (runs/sft/merged) and a fresh LoRA is
  trained on top, so the KL reference (adapter disabled) is the SFT model.
- reward = format_reward + rm_reward + form_reward, all on the raw output
  (a single emoji is judged as one emoji; tripling is display-only)
    format_reward: +1 if the raw output is 1-3 emoji only, else -2
    rm_reward:     reward-model score standardised *per emoji count* (mean/std of
                   the RM over the 5.2 candidates with the same count), so 1, 2 and
                   3 emoji are each compared only with their own kind; -2 if invalid
    form_reward:   steers the overall 1/2/3-emoji mix toward --form-target. Keeps an
                   EMA of the sampled mix p and pays lam * clip(log(target/p), -1, 1),
                   so under-used lengths get a bonus and over-used ones a penalty,
                   while the RM still decides which prompt gets which length.
- Every --eval-every steps, a fixed eval (200 held-out + adversarial prompts, greedy)
  logs format pass rate, the raw 1/2/3 mix, distinct emoji and mean RM score
  to runs/grpo/monitor.jsonl.

    python train_grpo.py --max-steps 50 --out runs/grpo_smoke     # smoke test
    python train_grpo.py                                         # full run
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "hf_cache"))

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from emoji_spec import finalize_reply, is_valid_reply, parse_emojis
from eval_format import generate
from train_rm import load_reward_model

DATA = Path(__file__).parent / "data"


def merged_sft(base: str, adapter: str, out: Path) -> str:
    if not (out / "config.json").exists():
        print(f"Merging {adapter} into {base} -> {out}")
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        model.save_pretrained(out)
        AutoTokenizer.from_pretrained(base).save_pretrained(out)
    return str(out)


def user_prompts(path):
    return [json.loads(l)["messages"][0]["content"] for l in open(path, encoding="utf-8")]


def completion_text(c) -> str:
    if isinstance(c, list):
        c = c[-1]
    if isinstance(c, dict):
        c = c.get("content", "")
    return (c or "").strip()


def n_emoji(reply: str) -> int:
    return len(parse_emojis(reply) or [])


def form_mix(replies) -> dict:
    lens = [n_emoji(r) for r in replies if is_valid_reply(r)]
    return {k: lens.count(k) / max(1, len(lens)) for k in (1, 2, 3)}


class Rewards:
    def __init__(self, rm, rm_stats: dict, target: dict, lam: float,
                 ema: float = 0.9, init_mix: dict | None = None):
        # rm_stats: {n_emoji: (mean, std)}. Each reply is standardised only against
        # replies with the same number of emoji, so the RM never prefers a length.
        self.rm, self.stats = rm, rm_stats
        self.target, self.lam, self.ema = target, lam, ema
        self.mix = dict(init_mix or target)

    def rm_std_units(self, prompts, replies):
        raw = self.rm.score(prompts, replies).tolist()
        return [(s - self.stats[n_emoji(r)][0]) / self.stats[n_emoji(r)][1]
                for s, r in zip(raw, replies)]

    def format_reward(self, prompts, completions, **kw):
        return [1.0 if is_valid_reply(completion_text(c)) else -2.0 for c in completions]

    def rm_reward(self, prompts, completions, **kw):
        users = [completion_text(p) for p in prompts]
        raws = [completion_text(c) for c in completions]
        out = [-2.0] * len(raws)
        idx = [i for i, r in enumerate(raws) if is_valid_reply(r)]
        if idx:
            scores = self.rm_std_units([users[i] for i in idx], [raws[i] for i in idx])
            for i, s in zip(idx, scores):
                out[i] = s
        return out

    def form_reward(self, prompts, completions, **kw):
        raws = [completion_text(c) for c in completions]
        batch = form_mix(raws)
        for k in self.mix:
            self.mix[k] = self.ema * self.mix[k] + (1 - self.ema) * batch[k]
        bonus = {k: self.lam * max(-1.0, min(1.0, math.log(self.target[k] / max(self.mix[k], 1e-3))))
                 for k in self.mix}
        return [bonus[n_emoji(r)] if is_valid_reply(r) else 0.0 for r in raws]


class Monitor(TrainerCallback):
    def __init__(self, tok, rewards: Rewards, prompts, every: int, log_path: Path):
        self.tok, self.rewards, self.prompts, self.every, self.log_path = tok, rewards, prompts, every, log_path

    def run(self, model, step):
        model.eval()
        side = self.tok.padding_side
        replies = generate(model, self.tok, self.prompts)
        self.tok.padding_side = side
        model.train()
        valid = [is_valid_reply(r) for r in replies]
        ok = [r for r, v in zip(replies, valid) if v]
        users = [p for p, v in zip(self.prompts, valid) if v]
        distinct = {e for r in ok for e in parse_emojis(r)}
        mix = form_mix(replies)
        row = {"step": step,
               "valid": sum(valid) / len(valid),
               "mix_greedy": mix,
               "mix_sampled_ema": dict(self.rewards.mix),
               "distinct_emoji": len(distinct),
               "rm_mean": sum(self.rewards.rm_std_units(users, ok)) / max(1, len(ok)),
               "samples": [[p, r, finalize_reply(r)] for p, r in list(zip(self.prompts, replies))[:8]]}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        fmt = lambda m: "/".join(f"{m[k]:.0%}" for k in (1, 2, 3))
        print(f"[monitor step {step}] valid {row['valid']:.1%}  mix(1/2/3) greedy {fmt(mix)}  "
              f"sampled {fmt(self.rewards.mix)}  distinct {row['distinct_emoji']}  "
              f"rm {row['rm_mean']:+.3f}", flush=True)

    def on_train_begin(self, args, state, control, model=None, **kw):
        self.run(model, 0)

    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step % self.every == 0:
            self.run(model, state.global_step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--sft-adapter", default="runs/sft/adapter")
    ap.add_argument("--rm", default="runs/rm/adapter")
    ap.add_argument("--out", default="runs/grpo")
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--form-target", default="0.25,0.40,0.35",
                    help="target share of 1,2,3-emoji raw replies")
    ap.add_argument("--form-lambda", type=float, default=0.5)
    args = ap.parse_args()
    target = dict(zip((1, 2, 3), map(float, args.form_target.split(","))))
    assert abs(sum(target.values()) - 1) < 1e-6, "--form-target must sum to 1"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "monitor.jsonl").unlink(missing_ok=True)

    policy_path = merged_sft(args.base, args.sft_adapter, Path("runs/sft/merged"))
    tok = AutoTokenizer.from_pretrained(policy_path)

    rm = load_reward_model(args.rm)
    calib = [json.loads(l) for l in open(DATA / "scores.jsonl", encoding="utf-8")]
    cp = [r["prompt"] for r in calib for _ in r["cands"]]
    cr = [c for r in calib for c in r["cands"]]
    s = rm.score(cp, cr)
    rm_stats = {}
    for k in (1, 2, 3):
        sk = torch.tensor([x for x, r in zip(s.tolist(), cr) if n_emoji(r) == k])
        rm_stats[k] = (sk.mean().item(), sk.std().item())
        print(f"RM calibration, {k} emoji: n={len(sk)} mean {rm_stats[k][0]:+.3f} std {rm_stats[k][1]:.3f}")
    sft_mix = form_mix([c["raw"] for r in map(json.loads, open(DATA / "candidates.jsonl", encoding="utf-8"))
                        for c in r["cands"]])
    rewards = Rewards(rm, rm_stats, target, args.form_lambda, init_mix=sft_mix)
    print(f"form target {target}; SFT sampled mix {sft_mix}")

    train_prompts = user_prompts(DATA / "sft_train.jsonl")
    random.Random(1).shuffle(train_prompts)
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": p}]} for p in train_prompts])

    held = user_prompts(DATA / "sft_eval.jsonl")[:200]
    adv = [json.loads(l)["prompt"] for l in open(DATA / "eval_adversarial.jsonl", encoding="utf-8")]
    monitor = Monitor(tok, rewards, held + adv, args.eval_every, out / "monitor.jsonl")

    completions_per_step = args.prompts_per_step * args.num_generations
    config = GRPOConfig(
        output_dir=str(out),
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=0.05,
        beta=args.beta,
        num_generations=args.num_generations,
        per_device_train_batch_size=completions_per_step // 4,
        gradient_accumulation_steps=4,
        max_completion_length=24,
        temperature=1.0,
        chat_template_kwargs={"enable_thinking": False},
        reward_weights=[1.0, 1.0, 1.0],
        bf16=True,
        logging_steps=5,
        save_steps=100,
        save_total_limit=8,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = GRPOTrainer(
        model=AutoModelForCausalLM.from_pretrained(policy_path, dtype=torch.bfloat16),
        reward_funcs=[rewards.format_reward, rewards.rm_reward, rewards.form_reward],
        args=config,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                               target_modules="all-linear", task_type="CAUSAL_LM"),
        callbacks=[monitor],
    )
    trainer.train()
    trainer.save_model(str(out / "adapter"))
    print(f"Saved GRPO adapter to {out / 'adapter'}")


if __name__ == "__main__":
    main()
