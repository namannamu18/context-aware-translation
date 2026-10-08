"""LoRA / QLoRA fine-tuning of TowerInstruct on English<->Chinese chat MT data (BMELD train).

This is the lightweight counterpart of the paper's fine-tuning step: the paper fine-tunes TowerInstruct-7B on chat
MT data with the conversation as context (scripts/training_tower_chat/w_context.py, add_chat_data.py) and trains it
with axolotl, giving `Unbabel/TowerInstruct-WMT24-Chat-7B`. Full fine-tuning of a 7B model does not fit on Kaggle's
free 2x T4 GPUs, so this script trains a LoRA adapter on top of a 4-bit quantised base model (QLoRA) instead.
Everything else follows the paper's data: one example per message, prompt = the `full_context` instruction in the
empty-system-prompt chat format (`chatml_empty_sys`, the format the paper's fine-tuned model is prompted with),
target = the reference translation, loss on the target only.

Runtime is bounded by --time_budget_min: training ends after one epoch or when the budget is used up, whichever is
first, and the learning rate is annealed against whichever limit is nearer.

  python scripts/finetune_lora.py --out_dir finetuned/zh_lora --time_budget_min 60
  # two GPUs (each holds its own 4-bit copy):  accelerate launch --num_processes 2 scripts/finetune_lora.py ...
  # quick check that everything runs:          python scripts/finetune_lora.py --max_examples 32 --max_steps 4 --eval_examples 16
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_mt_utils import TEMPLATES, build_prompts, load_jsonl  # noqa: E402

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def end_of_turn(template: str) -> str:
    return "<|eot_id|>" if template.startswith("llama3") else "<|im_end|>"


def encode_example(tok, prompt: str, target: str, end: str, max_len: int):
    """Tokenise prompt+target jointly (exactly how the text is tokenised at inference time, BOS included) and mask the
    prompt. If the prompt tokens are not a prefix of the joint tokenisation (a tokenizer that merges across the boundary)
    the target is tokenised separately instead. Returns (example, used_fallback), or (None, False) if it is too long."""
    head = tok(prompt)["input_ids"]
    full = tok(prompt + target + end)["input_ids"]
    fallback = full[: len(head)] != head
    if fallback:
        full = head + tok(target + end, add_special_tokens=False)["input_ids"]
    if len(full) > max_len or len(full) == len(head):
        return None, False
    return {"input_ids": full, "labels": [-100] * len(head) + full[len(head):]}, fallback


class ListDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            labels[i, :k] = torch.tensor(b["labels"])
            mask[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def build_dataset(tok, root: Path, dataset: str, template: str, use_context: bool, n_turns, max_examples, max_len, seed):
    recs = load_jsonl(root / "raw_data" / "mt" / f"{dataset}.zh" / "test.jsonl")
    prompts = build_prompts(recs, template, use_context, n_turns)
    assert len(prompts) == len(recs)
    idx = list(range(len(recs)))
    random.Random(seed).shuffle(idx)
    rows, skipped, fallbacks = [], 0, 0
    end = end_of_turn(template)
    for i in idx:
        if max_examples and len(rows) >= max_examples:
            break
        ex, fb = encode_example(tok, prompts[i], recs[i]["ref"], end, max_len)
        if ex is None:
            skipped += 1
        else:
            rows.append(ex)
            fallbacks += fb
    if not rows:
        raise SystemExit(f"No training example fits max_len={max_len} tokens ({skipped} skipped): check --max_len / --template.")
    if fallbacks:
        print(f"note: {fallbacks}/{len(rows)} examples were tokenised prompt and answer separately (tokenizer merges across the boundary)", flush=True)
    return rows, skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Unbabel/TowerInstruct-7B-v0.2")
    p.add_argument("--root_dir", default=str(Path(__file__).resolve().parent.parent))
    p.add_argument("--dataset", default="bmeld_train")
    p.add_argument("--eval_dataset", default="bmeld_dev", help="used only for the before/after dev loss")
    p.add_argument("--eval_examples", type=int, default=150, help="0 = skip the dev loss")
    p.add_argument("--context", choices=["full", "none"], default="full", help="train with the conversation as context (paper: w-context) or without")
    p.add_argument("--n_turns", type=int, default=None, help="limit the context to the last n messages (default: all, like full_context)")
    p.add_argument("--template", default="chatml_empty_sys", choices=[t for t in TEMPLATES if t != "no_template"])
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--max_len", type=int, default=768, help="examples longer than this many tokens are skipped")
    p.add_argument("--out_dir", default="finetuned/zh_lora")
    p.add_argument("--time_budget_min", type=float, default=None, help="stop (with an annealed learning rate) after this many minutes")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=4, help="per device")
    p.add_argument("--effective_batch", type=int, default=16, help="examples per optimizer step over all devices")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--load_in_4bit", choices=["auto", "yes", "no"], default="auto")
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--resume", action="store_true", help="continue from the last checkpoint in <out_dir>/checkpoints")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    cuda = torch.cuda.is_available()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    main_process = local_rank == 0
    log = (lambda *a: print(*a, flush=True)) if main_process else (lambda *a: None)

    # ---- precision / quantisation -------------------------------------------------------------------------------
    if args.dtype == "auto":
        dtype = "float32" if not cuda else ("bfloat16" if torch.cuda.get_device_capability()[0] >= 8 else "float16")
    else:
        dtype = args.dtype
    use_4bit = cuda if args.load_in_4bit == "auto" else args.load_in_4bit == "yes"
    if use_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            if args.load_in_4bit == "yes":
                raise
            log("bitsandbytes not installed: loading the base model without 4-bit quantisation")
            use_4bit = False
    torch_dtype = getattr(torch, dtype)
    log(f"base={args.base_model} dtype={dtype} 4bit={use_4bit} cuda={cuda} world_size={world}")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ---- data ------------------------------------------------------------------------------------------------------
    root = Path(args.root_dir)
    rows, skipped = build_dataset(tok, root, args.dataset, args.template, args.context == "full", args.n_turns, args.max_examples, args.max_len, args.seed)
    eval_rows = []
    if args.eval_examples:
        eval_rows, _ = build_dataset(tok, root, args.eval_dataset, args.template, args.context == "full", args.n_turns, args.eval_examples, args.max_len, args.seed)
    n_tokens = sum(len(r["input_ids"]) for r in rows)
    log(f"train examples: {len(rows)} (skipped {skipped}), {n_tokens/1e6:.2f}M tokens, mean {n_tokens/max(1,len(rows)):.0f} tokens/example; dev examples: {len(eval_rows)}")

    # ---- model -----------------------------------------------------------------------------------------------------
    kwargs = {"torch_dtype": torch_dtype}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch_dtype)
        kwargs["device_map"] = {"": local_rank}
    elif cuda:
        kwargs["device_map"] = {"": local_rank}
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs)
    model.config.use_cache = False
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    elif cuda:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    present = {n.split(".")[-1] for n, _ in model.named_modules()}
    targets = [t for t in TARGET_MODULES if t in present] or "all-linear"
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                            target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    for prm in model.parameters():            # fp16 training needs fp32 trainable weights (GradScaler)
        if prm.requires_grad and prm.dtype != torch.float32:
            prm.data = prm.data.float()
    if main_process:
        model.print_trainable_parameters()

    # ---- training ---------------------------------------------------------------------------------------------------
    budget_s = args.time_budget_min * 60 if args.time_budget_min else None
    grad_accum = max(1, args.effective_batch // (args.batch_size * world))
    out = Path(args.out_dir)
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant",           # replaced by the time-aware schedule below
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        fp16=(dtype == "float16" and cuda),
        bf16=(dtype == "bfloat16" and cuda),
        gradient_checkpointing=cuda,
        gradient_checkpointing_kwargs={"use_reentrant": False} if cuda else None,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        group_by_length=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        ddp_backend="gloo" if (world > 1 and not cuda) else None,   # gloo only for the CPU multi-process test
    )

    def synced_progress(local_value: float) -> float:
        """Rank 0's value on every rank, so that all ranks use the same learning rate and stop at the same step."""
        if world > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            t = torch.tensor([local_value], dtype=torch.float64, device="cuda" if cuda else "cpu")
            torch.distributed.broadcast(t, src=0)
            return float(t.item())
        return local_value

    class TimedTrainer(Trainer):
        t0 = None

        def elapsed(self):
            return 0.0 if self.t0 is None else time.time() - self.t0

        def create_scheduler(self, num_training_steps, optimizer=None):
            optimizer = optimizer or self.optimizer

            def factor(step):
                progress = step / max(1, num_training_steps)
                if budget_s:
                    progress = max(progress, synced_progress(self.elapsed() / budget_s))
                progress = min(1.0, progress)
                warm = min(1.0, (step + 1) / max(1, args.warmup_steps))
                return warm * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
            return self.lr_scheduler

    class BudgetCallback(TrainerCallback):
        def on_train_begin(self, a, state, control, **kw):
            trainer.t0 = time.time()

        def on_step_end(self, a, state, control, **kw):
            if budget_s and synced_progress(trainer.elapsed()) >= budget_s:
                log(f"time budget of {args.time_budget_min} min used up at step {state.global_step}: stopping")
                control.should_training_stop = True
            return control

    trainer = TimedTrainer(model=model, args=targs, train_dataset=ListDataset(rows), data_collator=Collator(tok.pad_token_id),
                           callbacks=[BudgetCallback()])

    info = {"base_model": args.base_model, "template": args.template, "context": args.context, "n_turns": args.n_turns, "dataset": args.dataset,
            "n_train_examples": len(rows), "skipped_too_long": skipped, "train_tokens_per_epoch": n_tokens, "dtype": dtype, "4bit": use_4bit,
            "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout, "targets": targets}, "lr": args.lr,
            "effective_batch": grad_accum * args.batch_size * world, "world_size": world, "time_budget_min": args.time_budget_min}
    if eval_rows:
        info["dev_loss_before"] = trainer.evaluate(eval_dataset=ListDataset(eval_rows), metric_key_prefix="dev")["dev_loss"]
        log(f"dev loss before training: {info['dev_loss_before']:.4f}")

    ckpt = None
    if args.resume and (out / "checkpoints").exists():
        cands = sorted((out / "checkpoints").glob("checkpoint-*"), key=lambda q: int(q.name.split("-")[-1]))
        ckpt = str(cands[-1]) if cands else None
    start = time.time()
    result = trainer.train(resume_from_checkpoint=ckpt)
    info["train_seconds"] = round(time.time() - start, 1)
    info["steps"] = trainer.state.global_step
    info["examples_seen"] = trainer.state.global_step * info["effective_batch"]
    info["train_loss"] = round(result.training_loss, 4)
    if eval_rows:
        info["dev_loss_after"] = trainer.evaluate(eval_dataset=ListDataset(eval_rows), metric_key_prefix="dev")["dev_loss"]
        log(f"dev loss after training: {info['dev_loss_after']:.4f}")

    if main_process:
        out.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(out))
        tok.save_pretrained(str(out))
        with open(out / "train_info.json", "w") as f:
            json.dump(info, f, indent=2)
        log("FINETUNE_DONE " + json.dumps(info))


if __name__ == "__main__":
    main()
