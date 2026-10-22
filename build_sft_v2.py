"""SFT v2, part 2: merge original + vocabulary data and cap over-used emoji.

- 5% of the vocabulary rows are held out as data/vocab_eval.jsonl.
- Vocabulary rows go in first (all rare emoji). Original rows follow in random
  order; a row is skipped if any of its emoji would exceed --cap of all emoji uses.
  The cap is measured against the final total, estimated iteratively.

    python build_sft_v2.py         # -> data/sft_v2_train.jsonl, data/vocab_eval.jsonl
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from emoji_spec import parse_emojis

DATA = Path(__file__).parent / "data"


def emojis(row):
    return parse_emojis(row["messages"][1]["content"])


def stats(name, rows):
    c = Counter(e for r in rows for e in emojis(r))
    tot = sum(c.values())
    mix = Counter(len(emojis(r)) for r in rows)
    print(f"{name}: {len(rows)} rows, {len(c)} distinct emoji, "
          f"mix 1/2/3 = {'/'.join(f'{mix[k] / len(rows):.0%}' for k in (1, 2, 3))}, "
          f"top: {' '.join(f'{e}{n / tot:.1%}' for e, n in c.most_common(8))}")


def capped(fixed, pool, cap, total_estimate):
    counts = Counter(e for r in fixed for e in emojis(r))
    kept = list(fixed)
    limit = cap * total_estimate
    for r in pool:
        es = emojis(r)
        add = Counter(es)
        if all(counts[e] + n <= limit for e, n in add.items()):
            kept.append(r)
            counts.update(add)
    return kept, sum(counts.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=0.02)
    ap.add_argument("--vocab-eval-ratio", type=float, default=0.05)
    args = ap.parse_args()

    orig = [json.loads(l) for l in open(DATA / "sft_train.jsonl", encoding="utf-8")]
    vocab = [json.loads(l) for l in open(DATA / "vocab_sft.jsonl", encoding="utf-8")]
    rng = random.Random(0)
    rng.shuffle(vocab)
    n_eval = int(len(vocab) * args.vocab_eval_ratio)
    vocab_eval, vocab_train = vocab[:n_eval], vocab[n_eval:]
    vocab_train = [{"messages": r["messages"]} for r in vocab_train]
    rng.shuffle(orig)

    stats("original", orig)
    stats("vocab", vocab_train)

    total = sum(len(emojis(r)) for r in orig + vocab_train)
    for _ in range(5):  # the kept total shrinks as rows are dropped; re-estimate
        kept, new_total = capped(vocab_train, orig, args.cap, total)
        if abs(new_total - total) < 50:
            break
        total = new_total
    rng.shuffle(kept)
    stats("sft_v2", kept)
    print(f"dropped {len(orig) + len(vocab_train) - len(kept)} original rows to respect the {args.cap:.0%} cap")

    with open(DATA / "sft_v2_train.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(DATA / "vocab_eval.jsonl", "w", encoding="utf-8") as f:
        for r in vocab_eval:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote data/sft_v2_train.jsonl ({len(kept)}) and data/vocab_eval.jsonl ({len(vocab_eval)})")


if __name__ == "__main__":
    main()
