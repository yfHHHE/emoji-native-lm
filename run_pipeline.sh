#!/usr/bin/env bash
# Wait for a running `gen_cli.py prompts` (PID $1, optional), then make sure
# prompts finish and run replies. Retries every 30 min on the usage limit.
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
LOG=data/pipeline.log
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1

if [ -n "$1" ]; then
  echo "$(date '+%F %T') waiting for PID $1" >> "$LOG"
  while tasklist //FI "PID eq $1" 2>/dev/null | grep -q "$1"; do sleep 60; done
fi

run_step() {  # run_step <name> <done-file>
  for attempt in $(seq 1 24); do
    [ -f "$2" ] && return 0
    echo "$(date '+%F %T') $1 attempt $attempt" >> "$LOG"
    "$PY" gen_cli.py "$1" >> "$LOG" 2>&1 && [ -f "$2" ] && return 0
    if tail -n 3 "$LOG" | grep -q "Usage limit"; then
      echo "$(date '+%F %T') usage limit, sleeping 30 min" >> "$LOG"
      sleep 1800
    else
      echo "$(date '+%F %T') $1 failed, not retrying" >> "$LOG"
      return 1
    fi
  done
  return 1
}

run_step prompts data/prompts.jsonl || exit 1
run_step replies data/sft_train.jsonl || exit 1
echo "$(date '+%F %T') pipeline done" >> "$LOG"
tail -n 3 "$LOG"
