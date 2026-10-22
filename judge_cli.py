"""Step 5.0 / 5.5: Opus pairwise judge between two models' generations (via `claude -p`).

Inputs are two files written by `eval_format.py --save` on the same prompts.
Every pair is judged twice with A/B swapped; disagreeing verdicts count as a tie,
which cancels position bias. A format-invalid reply loses without asking Opus.

    python judge_cli.py runs/sft/gen.jsonl runs/pilot/gen.jsonl
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from gen_cli import call_claude, run_jobs
from rubric import CRITERIA, rubric_hash

JUDGE_SYSTEM = f"""你是一个评审。对每条用户消息，比较回复 A 和回复 B，判断哪个作为回应更好。

{CRITERIA}
- 两个差不多好（或差不多差）时判 tie。"""

SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {"type": "string", "enum": ["A", "B", "tie"]}}},
    "required": ["verdicts"],
    "additionalProperties": False,
}


def load(path):
    return {(r["set"], r["prompt"]): r for r in map(json.loads, open(path, encoding="utf-8"))}


def make_job(items, swapped: bool):
    lines = []
    for i, (prompt, a, b) in enumerate(items):
        x, y = (b, a) if swapped else (a, b)
        lines.append(json.dumps({"i": i, "user": prompt, "A": x, "B": y}, ensure_ascii=False))
    text = (f"下面有 {len(items)} 组（JSON 行），每组彼此独立。按顺序对每一组给出判断，"
            f"verdicts 数组长度必须正好是 {len(items)}。\n\n" + "\n".join(lines))
    return lambda: call_claude(text, SCHEMA, JUDGE_SYSTEM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_a", help="gen.jsonl of the model under test")
    ap.add_argument("model_b", help="gen.jsonl of the reference model")
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=None, help="per-prompt verdicts jsonl")
    args = ap.parse_args()

    A, B = load(args.model_a), load(args.model_b)
    keys = [k for k in A if k in B]

    outcome = {}  # key -> "a" / "b" / "tie"
    to_judge = []
    for k in keys:
        va, vb = A[k]["valid"], B[k]["valid"]
        if va and vb:
            if A[k]["raw"] == B[k]["raw"]:
                outcome[k] = "tie"
            else:
                to_judge.append(k)
        else:
            outcome[k] = "a" if va else "b" if vb else "tie"

    chunks = [to_judge[i:i + args.chunk] for i in range(0, len(to_judge), args.chunk)]
    jobs, meta = [], []
    for ci, chunk in enumerate(chunks):
        items = [(k[1], A[k]["raw"], B[k]["raw"]) for k in chunk]
        digest = hashlib.sha1(json.dumps(items, ensure_ascii=False).encode()).hexdigest()[:12]
        for swapped in (False, True):
            jobs.append((f"j{rubric_hash(JUDGE_SYSTEM)}-{digest}-{int(swapped)}",
                         make_job(items, swapped)))
            meta.append((chunk, swapped))
    results = run_jobs(jobs, args.workers)

    votes = {}
    for (key, _), (chunk, swapped) in zip(jobs, meta):
        verdicts = results.get(key, {}).get("verdicts")
        if verdicts is None or len(verdicts) != len(chunk):
            print(f"[{key}] verdict count mismatch, treated as tie")
            verdicts = ["tie"] * len(chunk)
        for k, v in zip(chunk, verdicts):
            if swapped:
                v = {"A": "B", "B": "A"}.get(v, v)
            votes.setdefault(k, []).append({"A": "a", "B": "b"}.get(v, "tie"))
    for k, vs in votes.items():
        outcome[k] = vs[0] if len(vs) == 2 and vs[0] == vs[1] else "tie"

    print(f"\nA = {args.model_a}\nB = {args.model_b}")
    for s in ("held_out", "adversarial", None):
        ks = [k for k in keys if s is None or k[0] == s]
        c = Counter(outcome[k] for k in ks)
        n = len(ks)
        win_rate = (c["a"] + 0.5 * c["tie"]) / n
        print(f"{s or 'all':12s} n={n:4d}  A wins {c['a']:4d}  ties {c['tie']:4d}  B wins {c['b']:4d}"
              f"  -> A win rate (ties = half) {win_rate:.1%}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for k in keys:
                f.write(json.dumps({"set": k[0], "prompt": k[1], "a": A[k]["raw"],
                                    "b": B[k]["raw"], "outcome": outcome[k]},
                                   ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
