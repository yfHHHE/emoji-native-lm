"""SFT v2, part 1: widen the emoji vocabulary (via `claude -p`).

For every emoji used fewer than --min-uses times in data/sft_train.jsonl, Opus writes
conversations where that emoji is genuinely the most fitting reply:
  - non-flag emoji: 3 conversations, replies of exactly 1, 2 and 3 emoji
  - flags:          1 conversation, reply of 2 emoji
Every reply must contain the target emoji and pass the spec.

    python gen_vocab_cli.py            # -> data/vocab_sft.jsonl
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import emoji

from emoji_spec import is_valid_reply, normalize, parse_emojis
from gen_cli import call_claude, run_jobs

DATA = Path(__file__).parent / "data"
TONES = re.compile("[\U0001F3FB-\U0001F3FF]")

SYSTEM = """你在为一个“只用 1–3 个 emoji 回复”的聊天机器人构造训练数据，目标是让它学会使用更丰富的 emoji。

对每个目标 emoji，按要求写若干组对话（user 消息 + 机器人回复）：
- user 消息要像真实用户发给聊天机器人的话：大部分用中文，约三成用英文或中英混杂；长短、语气、话题都要多样。
- 这个 emoji 必须是真正贴切的回应，而不是硬塞进去的。冷门 emoji 就设计一个让它自然出现的话题（比如 🈂️ 可以是在日本看到的标志，♒ 可以是聊星座）。
- 回复只包含 emoji，必须包含目标 emoji，个数严格按要求；回复要像朋友那样回应消息，而不是把消息翻译成 emoji。
- 同一个目标 emoji 的几组对话话题要不同。"""

SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "emoji": {"type": "string"},
            "examples": {"type": "array", "items": {
                "type": "object",
                "properties": {"user": {"type": "string"}, "reply": {"type": "string"}},
                "required": ["user", "reply"], "additionalProperties": False}},
        },
        "required": ["emoji", "examples"], "additionalProperties": False}}},
    "required": ["items"],
    "additionalProperties": False,
}


def is_flag(e: str) -> bool:
    return all(0x1F1E6 <= ord(ch) <= 0x1F1FF for ch in e)


def target_emojis(min_uses: int):
    used = Counter(e for l in open(DATA / "sft_train.jsonl", encoding="utf-8")
                   for e in parse_emojis(json.loads(l)["messages"][1]["content"]))
    base = [e for e, d in emoji.EMOJI_DATA.items()
            if d.get("status") == emoji.STATUS["fully_qualified"] and not TONES.search(e)]
    return [e for e in base if used[e] < min_uses]


def make_job(targets, counts):
    spec = "\n".join(f"{e}：写 {len(counts)} 组，回复的 emoji 个数依次为 {'、'.join(map(str, counts))}"
                     for e in targets)
    text = (f"目标 emoji 共 {len(targets)} 个。items 按下面的顺序，每个目标 emoji 一项，"
            f"emoji 字段原样填写该 emoji：\n\n{spec}")
    return lambda: call_claude(text, SCHEMA, SYSTEM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-uses", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=30, help="target emoji per call")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="data/vocab_sft.jsonl")
    args = ap.parse_args()

    targets = target_emojis(args.min_uses)
    plain = [e for e in targets if not is_flag(e)]
    flags = [e for e in targets if is_flag(e)]
    print(f"targets: {len(plain)} non-flag (x3), {len(flags)} flags (x1)")

    jobs, plans = [], []
    for kind, lst, counts in (("v", plain, [1, 2, 3]), ("f", flags, [2])):
        for i in range(0, len(lst), args.chunk):
            chunk = lst[i:i + args.chunk]
            jobs.append((f"{kind}-{i:04d}-{len(chunk)}", make_job(chunk, counts)))
            plans.append((chunk, counts))
    results = run_jobs(jobs, args.workers)

    existing = {json.loads(l)["messages"][0]["content"] for l in open(DATA / "sft_train.jsonl", encoding="utf-8")}
    seen, rows = set(existing), []
    stats = Counter()
    for (key, _), (chunk, counts) in zip(jobs, plans):
        bare = lambda s: s.replace("️", "")
        items = {bare(it["emoji"]): it["examples"] for it in results.get(key, {}).get("items", [])}
        for e in chunk:
            exs = items.get(bare(e), [])
            if not exs:
                stats["emoji_missing"] += 1
            for ex, want in zip(exs, counts):
                user, reply = ex["user"].strip(), normalize(ex["reply"])
                found = parse_emojis(reply)
                if not user or user in seen:
                    stats["dup_or_empty"] += 1
                elif not is_valid_reply(reply):
                    stats["invalid_reply"] += 1
                elif bare(e) not in {bare(x) for x in found}:
                    stats["target_missing"] += 1
                else:
                    stats["kept"] += 1
                    stats[f"len_{len(found)}"] += 1
                    stats["len_mismatch"] += len(found) != want
                    seen.add(user)
                    rows.append({"target": e, "messages": [
                        {"role": "user", "content": user}, {"role": "assistant", "content": reply}]})

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    covered = len({r["target"] for r in rows})
    print(f"{dict(stats)}\ncovered {covered}/{len(targets)} target emoji; wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
