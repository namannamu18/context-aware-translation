"""Build *smoke-test* models so the full pipeline can be exercised on a CPU-only
machine with no access to the HuggingFace Hub.

!!! These models are NOT research models. Their translations and scores are
!!! meaningless and must never be reported as results. They exist only to
!!! check that every step of the pipeline (prompting, epsilon sampling, MBR,
!!! contrastive decoding, PCXMI, evaluation) runs end-to-end on en<->zh data.

Sub-commands
  lm     Train a tiny Llama-architecture causal LM + byte-level BPE tokenizer
         from scratch on BMELD *train* prompts (ChatML format, with and without
         context, empty-system and plain variants), using the same special
         tokens as TowerInstruct (<|im_start|>, <|im_end|> = EOS, <s> = BOS).
  comet  Create a randomly initialised COMET RegressionMetric (avg pooling, so
         `enable_context()` works) on top of a tiny XLM-R encoder with a locally
         trained tokenizer, saved as a normal COMET checkpoint
         (<dir>/checkpoints/model.ckpt + <dir>/hparams.yaml).

Examples
  python scripts/smoke_models.py lm --out smoke_models/tiny-chatml-lm --minutes 20
  python scripts/smoke_models.py comet --out smoke_models/tiny-comet
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_mt_utils import TEMPLATES, build_prompts, load_jsonl  # noqa: E402

SPECIAL = ["<s>", "<unk>", "<pad>", "<|im_start|>", "<|im_end|>"]


def _corpus(root: Path, dataset: str):
    recs = load_jsonl(root / "raw_data" / "mt" / f"{dataset}.zh" / "test.jsonl")
    return recs


def train_tokenizer(texts, vocab_size, out_dir):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator(texts, trainer)
    bos = tok.token_to_id("<s>")
    tok.post_processor = processors.TemplateProcessing(single="<s> $A", pair="<s> $A $B", special_tokens=[("<s>", bos)])
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<s>",
        eos_token="<|im_end|>",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=["<|im_start|>"],
    )
    hf.chat_template = (
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    hf.save_pretrained(out_dir)
    return hf


def cmd_lm(args):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    random.seed(0)
    torch.set_num_threads(args.threads)
    root = Path(args.root_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = _corpus(root, args.train_dataset)

    # training examples: every prompt format used by the pipeline for Tower-style models
    examples = []
    for template in ["chatml", "chatml_empty_sys"]:
        for use_context, n_turns in [(False, None), (True, args.max_context_turns)]:
            prompts = build_prompts(recs, template, use_context, n_turns)
            examples += [(p, r["ref"]) for p, r in zip(prompts, recs)]
    texts = [p + r for p, r in examples]
    if args.init_from:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.init_from)
        tok.save_pretrained(out)
    else:
        tok = train_tokenizer(texts, args.vocab_size, out)
    eos = tok.eos_token_id

    data = []
    for p, r in examples:
        p_ids = tok.encode(p)
        r_ids = tok.encode(r, add_special_tokens=False) + [eos]
        ids = (p_ids + r_ids)[-args.max_len :]
        n_r = min(len(r_ids), len(ids))
        labels = [-100] * (len(ids) - n_r) + ids[len(ids) - n_r :]
        data.append((ids, labels))

    cfg = LlamaConfig(
        vocab_size=len(tok),
        hidden_size=args.hidden,
        intermediate_size=args.hidden * 3,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.heads,
        max_position_embeddings=2048,
        bos_token_id=tok.bos_token_id,
        eos_token_id=eos,
        pad_token_id=tok.pad_token_id,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM.from_pretrained(args.init_from) if args.init_from else LlamaForCausalLM(cfg)
    cfg = model.config
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M, examples: {len(data)}")
    model.train()
    t0, step = time.time(), 0
    budget = args.minutes * 60
    log = []
    while time.time() - t0 < budget:
        random.shuffle(data)
        # length-bucketed batches
        for i in range(0, len(data), args.batch_size):
            batch = data[i : i + args.batch_size]
            L = max(len(x) for x, _ in batch)
            ids = torch.full((len(batch), L), tok.pad_token_id)
            lab = torch.full((len(batch), L), -100)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for j, (x, y) in enumerate(batch):
                ids[j, : len(x)] = torch.tensor(x)
                lab[j, : len(y)] = torch.tensor(y)
                att[j, : len(x)] = 1
            lr = args.lr * min(1.0, (step + 1) / 200)
            for g in opt.param_groups:
                g["lr"] = lr
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0:
                el = time.time() - t0
                print(f"step {step} loss {loss.item():.3f} ({el/60:.1f} min)", flush=True)
                log.append({"step": step, "loss": round(loss.item(), 4), "minutes": round(el / 60, 2)})
            if time.time() - t0 > budget:
                break
    model.eval()
    model.save_pretrained(out)
    with open(out / "SMOKE_TEST_MODEL_README.md", "w") as f:
        f.write(
            "# Smoke-test model -- NOT a research model\n\n"
            "Tiny Llama-architecture LM trained from scratch for a few CPU-minutes on BMELD train prompts "
            "(scripts/smoke_models.py). Used only to check that the pipeline runs end-to-end. "
            "Its outputs/scores are meaningless.\n\n" + json.dumps({"config": cfg.to_dict(), "train_log": log[-5:]}, indent=2)
        )
    print("saved", out)


def cmd_comet(args):
    import torch
    import yaml
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import XLMRobertaConfig, XLMRobertaTokenizerFast

    from comet.models import RegressionMetric

    torch.manual_seed(0)
    root = Path(args.root_dir)
    out = Path(args.out)
    enc_dir = out / "encoder"
    enc_dir.mkdir(parents=True, exist_ok=True)
    recs = _corpus(root, args.train_dataset)
    texts = [r["src"] for r in recs] + [r["ref"] for r in recs]

    # XLM-R style tokenizer (unigram, metaspace) trained locally
    tok = Tokenizer(models.Unigram())
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    specials = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    tok.train_from_iterator(texts, trainers.UnigramTrainer(vocab_size=args.vocab_size, special_tokens=specials, unk_token="<unk>"))
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", pair="<s> $A </s> </s> $B </s>", special_tokens=[("<s>", 0), ("</s>", 2)]
    )
    hf_tok = XLMRobertaTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", sep_token="</s>", cls_token="<s>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>")
    hf_tok.save_pretrained(enc_dir)
    XLMRobertaConfig(
        vocab_size=len(hf_tok),
        hidden_size=args.hidden,
        num_hidden_layers=args.layers,
        num_attention_heads=4,
        intermediate_size=args.hidden * 4,
        max_position_embeddings=514,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        type_vocab_size=1,
    ).save_pretrained(enc_dir)

    hparams = dict(
        nr_frozen_epochs=0.3,
        keep_embeddings_frozen=True,
        optimizer="AdamW",
        encoder_learning_rate=1e-6,
        learning_rate=1.5e-5,
        layerwise_decay=0.95,
        encoder_model="XLM-RoBERTa",
        pretrained_model=str(enc_dir.resolve()),
        pool="avg",
        layer="mix",
        layer_transformation="sparsemax",
        layer_norm=False,
        loss="mse",
        dropout=0.1,
        batch_size=16,
        train_data=[],
        validation_data=[],
        hidden_sizes=[64, 32],
        activations="Tanh",
        final_activation=None,
        load_pretrained_weights=False,
    )
    model = RegressionMetric(**hparams)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "hyper_parameters": hparams, "pytorch-lightning_version": "2.0.0"},
        ckpt_dir / "model.ckpt",
    )
    with open(out / "hparams.yaml", "w") as f:
        yaml.safe_dump({"class_identifier": "regression_metric", **hparams}, f)
    (out / "SMOKE_TEST_MODEL_README.md").write_text(
        "# Smoke-test COMET -- randomly initialised, NOT a quality metric\n\n"
        "Only used to exercise the COMET/context-COMET MBR code path offline.\n"
    )
    print("saved", ckpt_dir / "model.ckpt")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    lm = sub.add_parser("lm")
    lm.add_argument("--root_dir", default=".")
    lm.add_argument("--train_dataset", default="bmeld_train")
    lm.add_argument("--out", required=True)
    lm.add_argument("--vocab_size", type=int, default=8000)
    lm.add_argument("--hidden", type=int, default=256)
    lm.add_argument("--layers", type=int, default=4)
    lm.add_argument("--heads", type=int, default=4)
    lm.add_argument("--max_len", type=int, default=384)
    lm.add_argument("--max_context_turns", type=int, default=6)
    lm.add_argument("--batch_size", type=int, default=32)
    lm.add_argument("--lr", type=float, default=1e-3)
    lm.add_argument("--minutes", type=float, default=20)
    lm.add_argument("--threads", type=int, default=4)
    lm.add_argument("--init_from", default=None, help="Continue training an existing smoke model")
    c = sub.add_parser("comet")
    c.add_argument("--root_dir", default=".")
    c.add_argument("--train_dataset", default="bmeld_train")
    c.add_argument("--out", required=True)
    c.add_argument("--vocab_size", type=int, default=4000)
    c.add_argument("--hidden", type=int, default=128)
    c.add_argument("--layers", type=int, default=2)
    args = p.parse_args()
    {"lm": cmd_lm, "comet": cmd_comet}[args.cmd](args)


if __name__ == "__main__":
    main()
