"""Merge a LoRA adapter (scripts/finetune_lora.py) into its base model and save a standalone checkpoint, the
counterpart of the paper's released fine-tuned model (`Unbabel/TowerInstruct-WMT24-Chat-7B`). The merged model can
be used like any other model by the pipeline scripts (`MODEL=<out_dir>`), including vLLM.

The base model is loaded in 16-bit on the CPU (about 14 GB of RAM for a 7B model); the adapter was trained on a 4-bit
copy of the same weights, which is the usual QLoRA practice.

  python scripts/merge_lora.py --adapter finetuned/zh_lora --out_dir /kaggle/temp/tower_zh_merged
"""

import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="folder written by finetune_lora.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--base_model", default=None, help="default: the base model recorded in the adapter")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = p.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = args.base_model or json.load(open(Path(args.adapter) / "adapter_config.json"))["base_model_name_or_path"]
    print(f"merging {args.adapter} into {base}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(base).save_pretrained(args.out_dir)
    print(f"saved merged model to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
