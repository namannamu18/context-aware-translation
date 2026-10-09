"""Type-your-own-input version of the paper's primary system, for one conversation at a time:

    1. epsilon sampling of N candidates from the fine-tuned model's full-context prompt   (generate_candidates.py: T=0.7, min_p=0.02)
    2. context-aware COMET MBR picks the best candidate                                  (run_context_comet_mbr.py
       --use_context --context_source source --context_mt source --context_size 2)
    The paper submitted exactly this ("Contextual MBR re-ranking", the `mbr-source` column of paper_results) as its primary system.
    With verbose output you also get the greedy translations without / with context and the plain COMET MBR, the other columns
    of the paper's results table.

It reuses ChatTranslator (prompts identical to the pipeline's) and run_context_comet_mbr.run_mbr, and builds the COMET context
strings exactly like run_context_comet_mbr.add_context_across (sender = language of the message: the English speaker is the
"agent", the Chinese speaker the "customer", as in the BMELD data).

Differences to the batch pipeline: N is small by default (the paper uses 100; here 6), one conversation at a time, and no scores,
because a typed message has no reference translation.

Python:
    from translate_chat_pipeline import load_system, interactive_pipeline
    pt = load_system("Unbabel/TowerInstruct-7B-v0.2", adapter="finetuned/zh_lora")   # base model + LoRA adapter + COMET
    interactive_pipeline(pt)                                                          # type or paste "en: ..." / "zh: ..." lines
Command line:  python scripts/translate_chat_pipeline.py --adapter finetuned/zh_lora
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_context_comet_mbr as mbr_mod  # noqa: E402
from judge_chat import describe  # noqa: E402
from translate_chat import ChatTranslator, split_messages  # noqa: E402


class Turn(NamedTuple):
    lang: str          # language of the message ("en" / "zh"); also identifies the sender
    text: str          # the message as typed (source)
    comet_best: str    # its translation picked by plain COMET MBR (the "comet-best" column of the pipeline)
    final: str = ""    # the translation that was output (the paper's primary system); shown to the judge as context


class PipelineTranslator:
    def __init__(self, tr: ChatTranslator, comet_model: Optional[str] = "Unbabel/wmt22-comet-da", n_candidates: int = 6,
                 context_size: int = 2, context_mt: str = "source", batch_size: int = 16,
                 temperature: float = 0.7, min_p: float = 0.02, comet_device=None, judge=None):
        assert context_mt in ("comet-best", "source")
        self.tr = tr
        self.judge = judge          # optional ChatJudge (scripts/judge_chat.py): grades each final translation AFTER it was chosen
        self.n_candidates = n_candidates
        self.context_size = context_size
        self.context_mt = context_mt
        self.batch_size = batch_size
        self.temperature = temperature
        self.min_p = min_p
        self.comet = None
        self.utility = "chrf"
        if comet_model:
            try:
                self.comet = mbr_mod.load_comet(comet_model)
                self.utility = "comet"
            except Exception as e:  # no network / no checkpoint: chrF-MBR, as run_context_comet_mbr.py --utility chrf
                print(f"WARNING: COMET could not be loaded ({type(e).__name__}); falling back to chrF-MBR (not context-aware).")
        self.comet_device = comet_device if comet_device is not None else mbr_mod.DEVICE
        self.sep = self.comet.encoder.tokenizer.sep_token if self.comet is not None else "</s>"

    # -- COMET context strings: same rule as add_context_across, for the last message of a conversation ----------
    def context_strings(self, history: List[Turn], lang: str, text: str, candidates: List[str]) -> Tuple[str, List[str]]:
        window = history[max(0, len(history) - self.context_size):] if self.context_size > 0 else []
        src_ctx, out_ctx = [], []
        for t in window:
            same_sender = t.lang == lang
            mt = t.comet_best if self.context_mt == "comet-best" else t.text
            src_ctx.append(t.text if same_sender else mt)   # source side: same -> source, other -> context_mt
            out_ctx.append(mt if same_sender else t.text)   # output side: same -> context_mt, other -> source
        join = f" {self.sep} ".join
        return join(src_ctx + [text]), [join(out_ctx + [c]) for c in candidates]

    def _mbr(self, source: str, candidates: List[str], use_context: bool) -> int:
        n = len(candidates)
        if self.utility == "chrf":
            matrix = mbr_mod.run_mbr_chrf([source] * n, candidates, 1, n)
        else:
            matrix = mbr_mod.run_mbr(self.comet, [source] * n, candidates, 1, n, batch_size=self.batch_size,
                                     use_context=use_context, device_id=self.comet_device)
            self.comet.use_context = False  # as in run_context_comet_mbr.main
        return int(np.argmax(matrix[0]))

    def translate(self, text: str, lang: str, history: List[Turn], max_new_tokens: int = 128, full_report: bool = False) -> dict:
        """The final translation is r["final"] (the paper's primary system). full_report=True also computes the greedy
        translations and the plain COMET MBR (slower)."""
        tgt = "zh" if lang == "en" else "en"
        tr = self.tr
        ctx_prompt = tr.prompt(text, lang, tgt, [(t.lang, t.text) for t in history], True)
        greedy_ctx = greedy_no = mbr_plain = None
        if full_report:
            greedy_ctx = tr.generate([ctx_prompt], max_new_tokens)[0]
            greedy_no = tr.generate([tr.prompt(text, lang, tgt, None, False)], max_new_tokens)[0]
        candidates = [c for c in tr.sample(ctx_prompt, self.n_candidates, max_new_tokens, self.temperature, self.min_p) if c]
        if len(candidates) < 2:  # nothing to choose from
            final = candidates[0] if candidates else tr.generate([ctx_prompt], max_new_tokens)[0]
            mbr_plain = final
        else:
            if full_report or self.context_mt == "comet-best":
                mbr_plain = candidates[self._mbr(text, candidates, use_context=False)]
            src_c, outs_c = self.context_strings(history, lang, text, candidates)
            final = candidates[self._mbr(src_c, outs_c, use_context=True)]
        judged = None
        if self.judge is not None and self.judge.enabled:   # after the translation: the judge never influences the choice
            judged = self.judge.score(history, lang, text, final)
        return {"lang": lang, "tgt": tgt, "text": text, "final": final, "greedy_no_context": greedy_no, "greedy_context": greedy_ctx,
                "mbr": mbr_plain, "mbr_context": final, "candidates": candidates, "judge": judged,
                "turn": Turn(lang, text, mbr_plain if mbr_plain is not None else final, final)}

    def show(self, r: dict) -> None:
        print(f"\n[{r['lang']}->{r['tgt']}] {r['text']}")
        if r["greedy_context"] is None:
            print(f"   -> {r['final']}")
            if r.get("judge") is not None:
                print(describe(r["judge"], self.judge.judge_model))
            return
        print(f"   1 no context, greedy                    : {r['greedy_no_context']}")
        print(f"   2 with context, greedy                  : {r['greedy_context']}")
        print(f"   3 with context + COMET MBR              : {r['mbr']}")
        print(f"   4 with context + context-aware MBR (the paper's primary system): {r['final']}")
        if r.get("judge") is not None:
            print(describe(r["judge"], self.judge.judge_model))

    def run_conversation(self, messages: List[Tuple[str, str]], max_new_tokens: int = 128, verbose: bool = False) -> List[dict]:
        history, results = [], []
        for lang, text in messages:
            r = self.translate(text, lang, history, max_new_tokens, full_report=verbose)
            self.show(r)
            history.append(r["turn"])
            results.append(r)
        return results


def print_transcript(results: List[dict]) -> None:
    if not results:
        return
    print("\n=== translated conversation ===")
    for r in results:
        j = r.get("judge")
        mark = f"   (judge MQM {j['score']})" if j and j.get("score") is not None else ""
        print(f"[{r['lang']}] {r['text']}\n[{r['tgt']}] {r['final']}{mark}\n")
    scores = [r["judge"]["score"] for r in results if r.get("judge") and r["judge"].get("score") is not None]
    if scores:
        print(f"judge: mean MQM {sum(scores) / len(scores):.2f} over {len(scores)} of {len(results)} messages (0 = no errors found, more negative = worse)")


PROMPT_MARK = "\x01PROMPT"


def read_line() -> str:
    """input("> "), or, when the translator runs as a child process of a notebook (CAT_BRIDGE=1, see scripts/notebook_bridge.py), a marker line
    that tells the notebook to ask the user, then the answer from stdin."""
    if os.environ.get("CAT_BRIDGE") != "1":
        return input("> ")
    print(PROMPT_MARK, flush=True)
    line = sys.stdin.readline()
    if not line:
        raise EOFError
    return line.rstrip("\n")


def interactive_pipeline(pt: PipelineTranslator, max_new_tokens: int = 128, verbose: bool = False):
    print('Type or paste the conversation: messages starting with "en:" or "zh:" (several on one line are fine).')
    print('"reset" = new conversation, "verbose" = also show the greedy / plain-MBR translations, "judge" = judge on/off, "quit" = stop and print the transcript.')
    print(f"Each message is translated with the earlier messages as context: {pt.n_candidates} sampled candidates, best one picked by "
          f"{'context-aware COMET' if pt.utility == 'comet' else 'chrF'}.")
    if pt.judge is not None:
        print(f"After each translation the judge ({pt.judge.judge_model}) grades it (it only grades, it never changes the translation); type \"judge\" to switch it off/on.")
    history: List[Turn] = []
    results: List[dict] = []
    while True:
        try:
            line = read_line().strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() in {"quit", "exit", "q"}:
            break
        if line.lower() == "reset":
            print_transcript(results)
            history, results = [], []
            print("(new conversation)")
            continue
        if line.lower() == "judge":
            if pt.judge is None:
                print("(no judge: add the GEMINI_API_KEY secret and run the cell again)")
            else:
                pt.judge.enabled = not pt.judge.enabled
                print(f"(judge {'on' if pt.judge.enabled else 'off'})")
            continue
        if line.lower() == "verbose":
            verbose = not verbose
            print(f"(verbose {'on' if verbose else 'off'})")
            continue
        messages = split_messages(line)
        if not messages:
            print('Start the line with a language code, e.g. "en: Hello" or "zh: 你好".')
            continue
        for lang, text in messages:
            r = pt.translate(text, lang, history, max_new_tokens, full_report=verbose)
            pt.show(r)
            history.append(r["turn"])
            results.append(r)
    print_transcript(results)
    return results


def read_adapter_setup(adapter: str) -> dict:
    """What the adapter was trained with (written by scripts/finetune_lora.py): base model, prompt template, whether the context was used.
    This is how the translator is tied to the training run: nothing about the model has to be repeated by hand."""
    out, folder = {}, Path(adapter)
    if (folder / "adapter_config.json").exists():
        out["base_model"] = json.load(open(folder / "adapter_config.json")).get("base_model_name_or_path")
    if (folder / "train_info.json").exists():
        info = json.load(open(folder / "train_info.json"))
        out["template"], out["context"] = info.get("template"), info.get("context")
    return out


def load_system(base_model: Optional[str] = None, adapter: Optional[str] = None, comet_model: Optional[str] = "Unbabel/wmt22-comet-da",
                n_candidates: int = 6, dtype: str = "float16", device_map: Optional[str] = "auto", template: Optional[str] = None,
                **kwargs) -> PipelineTranslator:
    """Fine-tuned system: base model + LoRA adapter (merged on load) + COMET (+ optional judge=ChatJudge). With an adapter, the base model and the
    prompt template are taken from the adapter folder (what the training run used) unless given explicitly; without an adapter this is the base model."""
    setup = read_adapter_setup(adapter) if adapter else {}
    base_model = base_model or setup.get("base_model") or "Unbabel/TowerInstruct-7B-v0.2"
    template = template or setup.get("template") or "chatml_empty_sys"
    if adapter:
        print(f"adapter {adapter}: base model {base_model}, prompt format {template}")
        if setup.get("context") == "none":
            print("WARNING: this adapter was trained WITHOUT context, but the translator gives the earlier messages as context.")
    tr = ChatTranslator(base_model, dtype=dtype, device_map=device_map, template=template, adapter=adapter)
    return PipelineTranslator(tr, comet_model=comet_model, n_candidates=n_candidates, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default=None, help="default: the base model recorded in the adapter (else TowerInstruct-7B-v0.2)")
    p.add_argument("--adapter", default=None)
    p.add_argument("--comet_model", default="Unbabel/wmt22-comet-da", help="HF id or local .ckpt; 'none' = chrF-MBR")
    p.add_argument("--n_candidates", type=int, default=6)
    p.add_argument("--context_size", type=int, default=2)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device_map", default="auto")
    p.add_argument("--template", default=None, help="default: the one recorded in the adapter (else chatml_empty_sys)")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--judge", choices=["none", "gemini"], default="none", help="grade every translation afterwards (needs GEMINI_API_KEY; skipped with a message without it)")
    p.add_argument("--judge_model", default="gemini-2.5-flash")
    args = p.parse_args()
    judge = None
    if args.judge != "none":
        from judge_chat import ChatJudge

        judge = ChatJudge.create(judge_model=args.judge_model)
    pt = load_system(args.base_model, args.adapter, None if args.comet_model.lower() == "none" else args.comet_model, args.n_candidates,
                     args.dtype, args.device_map if args.device_map != "none" else None, args.template, context_size=args.context_size, judge=judge)
    interactive_pipeline(pt, args.max_new_tokens, args.verbose)


if __name__ == "__main__":
    main()
