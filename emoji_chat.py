"""Step 6.2: deployment wrapper - the only inference entry point.

reply(message):
  1. sample (temperature 1.0, top_p 0.9 - the deployment setting chosen in step 5)
  2. if the output is not 1-3 emoji only, resample (up to --retries times);
     then fall back to greedy, and finally to FALLBACK
  3. finalize: a single emoji is shown three times (😂 -> 😂😂😂)
The model was trained on single-turn messages, so each call sees only the latest message.

    python emoji_chat.py                         # interactive chat
    python emoji_chat.py --selftest              # wrapper on all eval prompts, reports stats
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

FALLBACK = "🙂"


class EmojiBot:
    def __init__(self, model_path="runs/final/model", temperature=1.0, top_p=0.9, retries=3):
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda").eval()
        self.temperature, self.top_p, self.retries = temperature, top_p, retries
        self.stats = Counter()

    @torch.no_grad()
    def _generate(self, messages, sample: bool):
        enc = self.tok([prompt_text(self.tok, m) for m in messages], return_tensors="pt",
                       padding=True, add_special_tokens=False).to(self.model.device)
        kw = ({"do_sample": True, "temperature": self.temperature, "top_p": self.top_p, "top_k": 0}
              if sample else {"do_sample": False})
        out = self.model.generate(**enc, max_new_tokens=24, pad_token_id=self.tok.pad_token_id, **kw)
        return [t.strip() for t in self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                                          skip_special_tokens=True)]

    def reply_batch(self, messages):
        raws = [None] * len(messages)
        todo = list(range(len(messages)))
        for attempt in range(1 + self.retries):
            if not todo:
                break
            for i, r in zip(todo, self._generate([messages[i] for i in todo], sample=True)):
                if is_valid_reply(r):
                    raws[i] = r
                    self.stats[f"ok_attempt_{attempt + 1}"] += 1
            todo = [i for i in todo if raws[i] is None]
        if todo:
            for i, r in zip(todo, self._generate([messages[i] for i in todo], sample=False)):
                raws[i] = r if is_valid_reply(r) else FALLBACK
                self.stats["greedy_fallback" if is_valid_reply(r) else "constant_fallback"] += 1
        return [finalize_reply(r) for r in raws], raws

    def reply(self, message: str) -> str:
        return self.reply_batch([message])[0][0]


def selftest(bot: EmojiBot):
    prompts = [json.loads(l)["messages"][0]["content"] for l in open("data/sft_eval.jsonl", encoding="utf-8")]
    prompts += [json.loads(l)["prompt"] for l in open("data/eval_adversarial.jsonl", encoding="utf-8")]
    torch.manual_seed(0)
    finals, raws = [], []
    for i in range(0, len(prompts), 32):
        f, r = bot.reply_batch(prompts[i:i + 32])
        finals += f
        raws += r
    shown_ok = sum(1 <= len(parse_emojis(f) or []) <= 3 and parse_emojis(f) is not None for f in finals)
    mix = Counter(len(parse_emojis(r)) for r in raws)
    print(f"prompts {len(prompts)}; shown replies that are 1-3 emoji only: {shown_ok}/{len(finals)}")
    print(f"attempts: {dict(sorted(bot.stats.items()))}")
    print(f"raw mix 1/2/3: {'/'.join(f'{mix[k] / len(raws):.0%}' for k in (1, 2, 3))}")
    for p, f in list(zip(prompts, finals))[:12]:
        print(f"  {p[:36]!r} -> {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/final/model")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    bot = EmojiBot(args.model)
    if args.selftest:
        selftest(bot)
        return
    print("Emoji bot ready. Type a message (Ctrl+C to quit).")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg:
            print("bot>", bot.reply(msg))


if __name__ == "__main__":
    main()
