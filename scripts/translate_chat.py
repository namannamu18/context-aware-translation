"""Interactive / quick context-aware chat translation (en<->zh and the other supported pairs).

Uses exactly the prompt formats of the pipeline (scripts/chat_mt_utils.py): the *no-context*
prompt and the *full-context* prompt, where the context is the previous utterances of the
conversation (both speakers, in their own languages). The model is loaded once with
HuggingFace transformers (device_map="auto" spreads it over all visible GPUs, e.g. 2x T4).

Python:
    from translate_chat import ChatTranslator
    tr = ChatTranslator("Unbabel/TowerInstruct-7B-v0.2", dtype="float16")
    history = []
    print(tr.translate("Hello, my order hasn't arrived.", "en", "zh", history))

Command line (type "en: <text>" or "zh: <text>", "reset" to start a new conversation, "quit" to stop):
    python scripts/translate_chat.py --model Unbabel/TowerInstruct-7B-v0.2 --dtype float16
    python scripts/translate_chat.py --model ... --quick_check bmeld_test --n_docs 3   # small check on real data
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_mt_utils import CODE_LANG_DICT, TEMPLATES, context_content, iter_conversations, load_jsonl, no_context_content  # noqa: E402


class ChatTranslator:
    def __init__(self, model_name_or_path: str, dtype: str = "float16", device_map: Optional[str] = "auto", template: str = "chatml",
                 repetition_penalty: float = 1.1):
        # repetition_penalty > 1 stops rare degenerate loops ("啊，啊，啊…") in greedy decoding; 1.0 = off
        self.repetition_penalty = repetition_penalty
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        kwargs = {"torch_dtype": getattr(torch, dtype)}
        if device_map and (torch.cuda.is_available() or device_map != "auto"):
            kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs).eval()
        self.template = TEMPLATES[template]
        self.eos_ids = [self.tokenizer.eos_token_id]
        for t in ["<|im_end|>", "<|eot_id|>"]:
            tid = self.tokenizer.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid != self.tokenizer.unk_token_id and tid not in self.eos_ids:
                self.eos_ids.append(tid)

    def prompt(self, src: str, src_lang: str, tgt_lang: str, history: Optional[List[Tuple[str, str]]] = None, use_context: bool = True) -> str:
        """history: previous utterances [(lang_code, text), ...] in their original languages."""
        s, t = CODE_LANG_DICT[src_lang], CODE_LANG_DICT[tgt_lang]
        if use_context and history:
            content = context_content(src, s, t, [text for _, text in history])
        else:
            content = no_context_content(src, s, t)
        return self.template.format(content=content)

    def generate(self, prompts: List[str], max_new_tokens: int = 128) -> List[str]:
        outs = []
        for p in prompts:
            enc = self.tokenizer(p, return_tensors="pt").to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=self.repetition_penalty,
                    eos_token_id=self.eos_ids,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos_ids[0],
                )
            outs.append(self.tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        return outs

    def translate(self, src: str, src_lang: str, tgt_lang: str, history=None, use_context: bool = True, max_new_tokens: int = 128) -> str:
        return self.generate([self.prompt(src, src_lang, tgt_lang, history, use_context)], max_new_tokens)[0]


def quick_check(tr: ChatTranslator, root: Path, dataset: str, n_docs: int, n_show: int = 6, max_new_tokens: int = 128):
    """Translate the first n_docs conversations of a prepared dataset (both directions), with and
    without context, and report chrF/BLEU per direction and a few examples."""
    from sacrebleu.metrics import BLEU, CHRF

    xx = "zh"
    records = load_jsonl(root / "raw_data" / "mt" / f"{dataset}.{xx}" / "test.jsonl")
    convos = list(iter_conversations(records))[:n_docs]
    rows = []
    for convo in convos:
        history = []
        for r in convo:
            for use_ctx in (False, True):
                p = tr.prompt(r["src"], r["source_language"], r["target_language"], history, use_ctx)
                rows.append((r, use_ctx, p))
            history.append((r["source_language"], r["src"]))
    hyps = tr.generate([p for _, _, p in rows], max_new_tokens)
    results = {}
    for lp in sorted({f"{r['source_language']}-{r['target_language']}" for r, _, _ in rows}):
        tgt = lp.split("-")[1]
        for use_ctx in (False, True):
            sel = [(r, h) for (r, c, _), h in zip(rows, hyps) if c == use_ctx and f"{r['source_language']}-{r['target_language']}" == lp]
            H, R = [h for _, h in sel], [r["ref"] for r, _ in sel]
            key = f"{lp} {'full_context' if use_ctx else 'no_context'}"
            results[key] = {
                "segments": len(H),
                "chrF": round(CHRF().corpus_score(H, [R]).score, 1),
                "BLEU": round(BLEU(tokenize="zh" if tgt == "zh" else "13a").corpus_score(H, [R]).score, 1),
            }
    print("\n=== Scores (higher is better) ===")
    for k, v in results.items():
        print(f"{k:24s} {v}")
    print("\n=== Examples (source / reference / no-context / full-context) ===")
    by_key = {}
    for (r, c, _), h in zip(rows, hyps):
        by_key.setdefault(id(r), {"r": r})[c] = h
    for i, d in enumerate(list(by_key.values())[:n_show]):
        r = d["r"]
        print(f"\n[{r['source_language']}->{r['target_language']}] {r['src']}")
        print(f"   reference    : {r['ref']}")
        print(f"   no context   : {d[False]}")
        print(f"   with context : {d[True]}")
    return results


def interactive(tr: ChatTranslator, max_new_tokens: int = 128, show_no_context: bool = True):
    print('Type "en: <English text>" or "zh: <Chinese text>"; "reset" = new conversation; "quit" = stop.')
    history = []
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() in {"quit", "exit", "q"}:
            break
        if line.lower() == "reset":
            history = []
            print("(new conversation)")
            continue
        if ":" not in line or line.split(":", 1)[0].strip().lower() not in CODE_LANG_DICT:
            print('Start the line with a language code, e.g. "en: Hello" or "zh: 你好".')
            continue
        lang, text = line.split(":", 1)
        lang, text = lang.strip().lower(), text.strip()
        tgt = "zh" if lang == "en" else "en"
        out = tr.translate(text, lang, tgt, history, use_context=True, max_new_tokens=max_new_tokens)
        print(f"  [{lang}->{tgt}, with context] {out}")
        if show_no_context and history:
            print(f"  [{lang}->{tgt}, no context]   {tr.translate(text, lang, tgt, None, use_context=False, max_new_tokens=max_new_tokens)}")
        history.append((lang, text))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Unbabel/TowerInstruct-7B-v0.2")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device_map", default="auto")
    p.add_argument("--template", default="chatml", choices=["chatml", "chatml_empty_sys", "llama3_empty_sys"])
    p.add_argument("--root_dir", default=str(Path(__file__).resolve().parent.parent))
    p.add_argument("--quick_check", default=None, help="Dataset for a small check, e.g. bmeld_test")
    p.add_argument("--n_docs", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    args = p.parse_args()
    tr = ChatTranslator(args.model, args.dtype, args.device_map, args.template, args.repetition_penalty)
    if args.quick_check:
        quick_check(tr, Path(args.root_dir), args.quick_check, args.n_docs, max_new_tokens=args.max_new_tokens)
    else:
        interactive(tr, args.max_new_tokens)


if __name__ == "__main__":
    main()
