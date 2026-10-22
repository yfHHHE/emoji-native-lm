"""Shared helpers for Message Batches: submit once, resume by saved batch id."""
import time
from pathlib import Path

import anthropic

MODEL = "claude-opus-5-5"
client = anthropic.Anthropic()


def submit_or_resume(requests, id_file: Path) -> str:
    """Submit `requests` unless `id_file` already holds a batch id; return the id."""
    if id_file.exists():
        batch_id = id_file.read_text().strip()
        print(f"Resuming batch {batch_id}")
        return batch_id
    batch = client.messages.batches.create(requests=requests)
    id_file.parent.mkdir(parents=True, exist_ok=True)
    id_file.write_text(batch.id)
    print(f"Submitted batch {batch.id} with {len(requests)} requests")
    return batch.id


def wait(batch_id: str, poll_seconds: int = 60):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        print(f"[{batch.processing_status}] processing={c.processing} "
              f"succeeded={c.succeeded} errored={c.errored}", flush=True)
        if batch.processing_status == "ended":
            return batch
        time.sleep(poll_seconds)


def iter_texts(batch_id: str):
    """Yield (custom_id, text) for succeeded, non-refused results."""
    skipped = {"errored": 0, "canceled": 0, "expired": 0, "refusal": 0}
    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            skipped[result.result.type] += 1
            continue
        msg = result.result.message
        if msg.stop_reason == "refusal":
            skipped["refusal"] += 1
            continue
        text = next((b.text for b in msg.content if b.type == "text"), "")
        yield result.custom_id, text
    print(f"Skipped: {skipped}")
