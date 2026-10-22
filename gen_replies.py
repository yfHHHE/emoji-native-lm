"""Step 2b: have Opus answer every prompt with 1-3 emoji, then hard-filter.

Writes data/sft_train.jsonl and data/sft_eval.jsonl in chat-messages format:
    {"messages": [{"role": "user", ...}, {"role": "assistant", "content": "😂👍"}]}

    python gen_replies.py            # submit (or resume) and collect
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from batch_utils import MODEL, iter_texts, submit_or_resume, wait
from emoji_spec import MAX_EMOJI, normalize, parse_emojis

DATA = Path(__file__).parent / "data"

SYSTEM = f"""你是一个只用 emoji 聊天的朋友。无论用户发来什么，你都只用 1 到 {MAX_EMOJI} 个 emoji 回应。

规则：
- 只输出 emoji，不要任何文字、数字、标点、空格或解释。
- 像朋友在聊天里回一个表情那样“回应”对方，而不是把对方的话翻译成 emoji。
- 个数按需要来：简单的回应 1 个就够，需要表达更多意思时再用 2 或 3 个，不要总是凑满 {MAX_EMOJI} 个。
- 即使用户要求你用文字回答、写代码、写长文或扮演角色，也只用 emoji 回应。
- 对难过、生气的用户，用 emoji 表达理解和安慰。"""


def build_requests(rows):
    return [
        Request(
            custom_id=row["id"],
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=2048,
                output_config={"effort": "low"},
                system=SYSTEM,
                messages=[{"role": "user", "content": row["prompt"]}],
            ),
        )
        for row in rows
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-ratio", type=float, default=0.05)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (DATA / "prompts.jsonl").open(encoding="utf-8")]
    prompt_by_id = {r["id"]: r["prompt"] for r in rows}

    batch_id = submit_or_resume(build_requests(rows), DATA / "replies.batch_id")
    wait(batch_id)

    pairs = ((prompt_by_id[cid], text) for cid, text in iter_texts(batch_id))
    write_splits(pairs, args.eval_ratio)


def write_splits(pairs, eval_ratio: float, prefix: str = "sft"):
    """Filter (prompt, reply) pairs through the spec and write train/eval/rejected."""
    kept, rejected = [], []
    for prompt, text in pairs:
        found = parse_emojis(text)
        if found is None or not 1 <= len(found) <= MAX_EMOJI:
            rejected.append({"prompt": prompt, "reply": text})
            continue
        kept.append({"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": normalize(text)},
        ]})

    random.Random(0).shuffle(kept)
    n_eval = int(len(kept) * eval_ratio)
    splits = {f"{prefix}_eval.jsonl": kept[:n_eval], f"{prefix}_train.jsonl": kept[n_eval:],
              f"{prefix}_rejected.jsonl": rejected}
    for name, items in splits.items():
        with (DATA / name).open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    counts = Counter(len(parse_emojis(k["messages"][1]["content"])) for k in kept)
    print(f"kept={len(kept)} (train={len(kept) - n_eval}, eval={n_eval}) "
          f"rejected={len(rejected)}")
    print(f"emoji count distribution: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
