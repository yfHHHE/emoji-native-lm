"""Step 5.2: Opus scores every distinct valid candidate 1-10 (via `claude -p`).

Only prompts with >= 2 distinct valid candidates are scored: the reward model
learns from comparisons within a prompt, so a lone candidate teaches nothing.

    python score_cli.py            # data/candidates.jsonl -> data/scores.jsonl
"""
import argparse
import hashlib
import json
from pathlib import Path

from emoji_spec import normalize, parse_emojis
from gen_cli import call_claude, run_jobs
from rubric import CRITERIA, rubric_hash

DATA = Path(__file__).parent / "data"

SCORE_SYSTEM = f"""你是一个评审，给候选回复打分。

{CRITERIA}

打分：1–10 的整数。10 = 非常贴切、让人会心一笑；7 = 不错；5 = 一般，勉强对得上；3 = 牵强或万能 emoji；1 = 答非所问或错误。
同一条消息下的候选要拉开差距，分数要能体现它们之间的优劣；确实一样好时可以同分。"""

SCHEMA = {
    "type": "object",
    "properties": {"scores": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}},
    "required": ["scores"],
    "additionalProperties": False,
}


def make_job(group):
    lines = [json.dumps({"i": i, "user": g["prompt"], "candidates": g["finals"]}, ensure_ascii=False)
             for i, g in enumerate(group)]
    shape = ", ".join(str(len(g["finals"])) for g in group)
    text = (f"下面有 {len(group)} 条用户消息（JSON 行），每条附带若干候选回复。"
            f"按顺序给每条消息的每个候选打分。scores 是 {len(group)} 个数组，"
            f"第 i 个数组按顺序对应第 i 条消息的候选，长度依次为：{shape}。\n\n" + "\n".join(lines))
    return lambda: call_claude(text, SCHEMA, SCORE_SYSTEM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/candidates.jsonl")
    ap.add_argument("--out", default="data/scores.jsonl")
    ap.add_argument("--chunk", type=int, default=15, help="prompts per call")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only-with-single", action="store_true",
                    help="score only prompts that have a single-emoji candidate")
    ap.add_argument("--merge-into", default=None,
                    help="existing scores.jsonl whose rows for these prompts get replaced")
    args = ap.parse_args()

    # Candidates are scored as the model wrote them (raw), not after the
    # single-emoji tripling, so 😂 is judged as one emoji.
    groups = []
    for row in map(json.loads, open(args.inp, encoding="utf-8")):
        raws = list(dict.fromkeys(normalize(c["raw"]) for c in row["cands"] if c["valid"]))
        if len(raws) < 2:
            continue
        if args.only_with_single and not any(len(parse_emojis(r)) == 1 for r in raws):
            continue
        groups.append({"prompt": row["prompt"], "finals": raws})
    chunks = [groups[i:i + args.chunk] for i in range(0, len(groups), args.chunk)]
    print(f"{len(groups)} prompts to score in {len(chunks)} calls")

    tag = rubric_hash(SCORE_SYSTEM)
    digest = lambda c: hashlib.sha1(json.dumps(c, ensure_ascii=False).encode()).hexdigest()[:10]
    jobs = [(f"s{tag}-{ci:04d}-{digest(c)}", make_job(c)) for ci, c in enumerate(chunks)]
    results = run_jobs(jobs, args.workers)

    kept, dropped = [], 0
    for (key, _), chunk in zip(jobs, chunks):
        scores = results.get(key, {}).get("scores")
        if scores is None or len(scores) != len(chunk):
            dropped += len(chunk)
            continue
        for g, s in zip(chunk, scores):
            if len(s) != len(g["finals"]):
                dropped += 1
                continue
            kept.append({"prompt": g["prompt"], "cands": g["finals"],
                         "scores": [min(10, max(1, int(x))) for x in s]})

    rows = kept
    if args.merge_into:
        new = {r["prompt"] for r in kept}
        old = [r for r in map(json.loads, open(args.merge_into, encoding="utf-8"))
               if r["prompt"] not in new]
        rows = old + kept
        print(f"merged: {len(old)} existing rows kept, {len(kept)} replaced/added")
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"scored prompts: kept {len(kept)}, dropped {dropped}; wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
