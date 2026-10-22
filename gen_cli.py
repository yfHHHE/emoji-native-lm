"""Step 2 via Claude Code headless mode (`claude -p`): uses your Claude
subscription instead of an API key. Same outputs as gen_prompts.py / gen_replies.py.

Every call's result is cached in data/cli_cache/, so an interrupted run (or one
that hit the subscription usage limit) picks up where it stopped.

    python gen_cli.py prompts                  # -> data/prompts.jsonl
    python gen_cli.py replies                  # -> data/sft_{train,eval,rejected}.jsonl
"""
import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gen_prompts import INSTRUCTION, SCHEMA as PROMPTS_SCHEMA, STYLES, TOPICS, write_prompts
from gen_replies import SYSTEM as REPLY_SYSTEM, write_splits

DATA = Path(__file__).parent / "data"
CACHE = DATA / "cli_cache"
MODEL = "claude-opus-5-5"
CLAUDE = shutil.which("claude")

REPLIES_SCHEMA = {
    "type": "object",
    "properties": {"replies": {"type": "array", "items": {"type": "string"}}},
    "required": ["replies"],
    "additionalProperties": False,
}


class UsageLimitError(RuntimeError):
    pass


def call_claude(prompt: str, schema: dict, system: str, timeout: int = 600) -> dict:
    """One headless call with no tools; returns the structured output dict."""
    cmd = [CLAUDE, "-p", "--model", MODEL, "--effort", "low", "--tools", "",
           "--no-session-persistence", "--output-format", "json",
           "--system-prompt", system, "--json-schema", json.dumps(schema)]
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout)
    out = json.loads(proc.stdout)
    if out.get("is_error"):
        msg = str(out.get("result", ""))
        if "limit" in msg.lower() or "authenticate" in msg.lower():
            raise UsageLimitError(msg)
        raise RuntimeError(msg)
    if out.get("structured_output") is not None:
        return out["structured_output"]
    return json.loads(out["result"])


def run_jobs(jobs, workers: int):
    """jobs: list of (cache_key, fn). Runs uncached ones; returns {key: result}."""
    CACHE.mkdir(parents=True, exist_ok=True)
    results, todo = {}, []
    for key, fn in jobs:
        path = CACHE / f"{key}.json"
        if path.exists():
            results[key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            todo.append((key, fn))
    print(f"{len(results)} cached, {len(todo)} to run")

    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fn): key for key, fn in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            key = futures[fut]
            try:
                results[key] = fut.result()
            except UsageLimitError as e:
                pool.shutdown(cancel_futures=True)
                hint = ("Log in first: run `claude`, then `/login`."
                        if "authenticate" in str(e).lower()
                        else "Usage limit hit; re-run after it resets.")
                raise SystemExit(f"Stopped: {e}\n{hint} Finished calls are cached.")
            except Exception as e:  # noqa: BLE001 - log and keep going
                print(f"[{key}] failed: {e}")
                continue
            (CACHE / f"{key}.json").write_text(
                json.dumps(results[key], ensure_ascii=False), encoding="utf-8")
            print(f"[{i}/{len(todo)}] {key}", flush=True)
    return results


def cmd_prompts(args):
    jobs = []
    for r in range(args.rounds):
        for ti, topic in enumerate(TOPICS):
            for si, style in enumerate(STYLES):
                text = INSTRUCTION.format(topic=topic, style=style, n=args.per_request)
                jobs.append((f"p-r{r}-t{ti}-s{si}",
                             lambda text=text: call_claude(text, PROMPTS_SCHEMA, "你在构造训练数据。")))
    results = run_jobs(jobs, args.workers)
    write_prompts((key, res["messages"]) for key, res in results.items())


def cmd_replies(args):
    rows = [json.loads(l) for l in (DATA / "prompts.jsonl").open(encoding="utf-8")]
    chunks = [rows[i:i + args.chunk] for i in range(0, len(rows), args.chunk)]

    def make_job(chunk):
        numbered = "\n".join(json.dumps({"i": i, "message": r["prompt"]}, ensure_ascii=False)
                             for i, r in enumerate(chunk))
        text = (f"下面是 {len(chunk)} 条彼此独立的用户消息（JSON 行）。按顺序对每一条分别给出你的回应，"
                f"replies 数组长度必须正好是 {len(chunk)}，第 i 个元素就是对第 i 条消息的回应。\n\n{numbered}")
        return lambda: call_claude(text, REPLIES_SCHEMA, REPLY_SYSTEM)

    jobs = [(f"r-{c[0]['id']}", make_job(c)) for c in chunks]
    results = run_jobs(jobs, args.workers)

    pairs = []
    for (key, _), chunk in zip(jobs, chunks):
        replies = results.get(key, {}).get("replies")
        if replies is None or len(replies) != len(chunk):
            print(f"[{key}] reply count mismatch, chunk dropped")
            continue
        pairs.extend((row["prompt"], reply) for row, reply in zip(chunk, replies))
    write_splits(pairs, args.eval_ratio)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--per-request", type=int, default=25)
    p.add_argument("--workers", type=int, default=3)
    r = sub.add_parser("replies")
    r.add_argument("--chunk", type=int, default=40, help="user messages per call")
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--eval-ratio", type=float, default=0.05)
    args = ap.parse_args()
    if CLAUDE is None:
        raise SystemExit("`claude` CLI not found on PATH")
    {"prompts": cmd_prompts, "replies": cmd_replies}[args.cmd](args)


if __name__ == "__main__":
    main()
