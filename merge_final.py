"""Step 6.1: bake the chosen pipeline (Qwen3-1.7B + SFT v2 + DPO2) into one model.

runs/sft_v2/merged already holds base + SFT v2; this merges the DPO2 LoRA on top.

    python merge_final.py            # -> runs/final/model
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "hf_cache"))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/sft_v2/merged")
    ap.add_argument("--adapter", default="runs/dpo2/adapter")
    ap.add_argument("--out", default="runs/final/model")
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print(f"Saved merged model to {args.out}")


if __name__ == "__main__":
    main()
