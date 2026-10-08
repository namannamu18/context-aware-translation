"""Type-your-own-input version of the paper's full method (the pipeline of run_zh_pipeline.sh, for one
conversation at a time):

    1. greedy translation without context and with context     (generate_translations.py)
    2. epsilon sampling of N candidates from the context prompt (generate_candidates.py: T=0.7, min_p=0.02)
    3. COMET MBR picks the best candidate                       (run_context_comet_mbr.py, "comet_eps")
    4. context-aware COMET MBR picks the best candidate         (run_context_comet_mbr.py,
       --use_context --context_source source --context_mt comet-best)

It reuses ChatTranslator (prompts identical to the pipeline's) and run_context_comet_mbr.run_mbr, and builds the
COMET context strings exactly like run_context_comet_mbr.add_context_across (sender = language of the message:
the English speaker is the "agent", the Chinese speaker the "customer", as in the BMELD data).

Differences to the batch pipeline: N is small by default (the paper uses 100), one conversation at a time, and
no scores, because a typed message has no reference translation.

Python (after `tr = ChatTranslator(...)`):
    from translate_chat_pipeline import PipelineTranslator, interactive_pipeline
    pt = PipelineTranslator(tr, comet_model="Unbabel/wmt22-comet-da", n_candidates=16)
    pt.run_conversation([("en", "Hi, I bought a coat but it is too small."), ("zh", "那我可以换一件大一点的吗？")])
    interactive_pipeline(pt)      # type or paste "en: ..." / "zh: ..." lines
"""

import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_context_comet_mbr as mbr_mod  # noqa: E402
from translate_chat import ChatTranslator, split_messages  # noqa: E402


class Turn(NamedTuple):
    lang: str          # language of the message ("en" / "zh"); also identifies the sender
    text: str          # the message as typed (source)
    comet_best: str    # its translation picked by plain COMET MBR (the "comet-best" column of the pipeline)


class PipelineTranslator:
    def __init__(self, tr: ChatTranslator, comet_model: str = "Unbabel/wmt22-comet-da", n_candidates: int = 16,
                 context_size: int = 2, context_mt: str = "comet-best", batch_size: int = 16,
                 temperature: float = 0.7, min_p: float = 0.02, comet_device=None):
        assert context_mt in ("comet-best", "source")
        self.tr = tr
        self.n_candidates = n_candidates
        self.context_size = context_size
        self.context_mt = context_mt
        self.batch_size = batch_size
        self.temperature = temperature
        self.min_p = min_p
        self.comet = mbr_mod.load_comet(comet_model)
        self.comet_device = comet_device if comet_device is not None else mbr_mod.DEVICE
        self.sep = self.comet.encoder.tokenizer.sep_token

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
        matrix = mbr_mod.run_mbr(self.comet, [source] * n, candidates, 1, n, batch_size=self.batch_size,
                                 use_context=use_context, device_id=self.comet_device)
        self.comet.use_context = False  # as in run_context_comet_mbr.main
        return int(np.argmax(matrix[0]))

    def translate(self, text: str, lang: str, history: List[Turn], max_new_tokens: int = 128) -> dict:
        tgt = "zh" if lang == "en" else "en"
        tr = self.tr
        ctx_prompt = tr.prompt(text, lang, tgt, [(t.lang, t.text) for t in history], True)
        greedy_ctx = tr.generate([ctx_prompt], max_new_tokens)[0]
        greedy_no = tr.generate([tr.prompt(text, lang, tgt, None, False)], max_new_tokens)[0]
        candidates = [c for c in tr.sample(ctx_prompt, self.n_candidates, max_new_tokens, self.temperature, self.min_p) if c]
        if len(candidates) < 2:  # nothing to choose from
            mbr_plain = mbr_ctx = greedy_ctx
        else:
            mbr_plain = candidates[self._mbr(text, candidates, use_context=False)]
            src_c, outs_c = self.context_strings(history, lang, text, candidates)
            mbr_ctx = candidates[self._mbr(src_c, outs_c, use_context=True)]
        return {"lang": lang, "tgt": tgt, "text": text, "greedy_no_context": greedy_no, "greedy_context": greedy_ctx,
                "mbr": mbr_plain, "mbr_context": mbr_ctx, "candidates": candidates,
                "turn": Turn(lang, text, mbr_plain)}

    def show(self, r: dict) -> None:
        print(f"\n[{r['lang']}->{r['tgt']}] {r['text']}")
        print(f"   1 no context, greedy                    : {r['greedy_no_context']}")
        print(f"   2 with context, greedy                  : {r['greedy_context']}")
        print(f"   3 with context + COMET MBR              : {r['mbr']}")
        print(f"   4 with context + context-aware MBR (full method): {r['mbr_context']}")

    def run_conversation(self, messages: List[Tuple[str, str]], max_new_tokens: int = 128) -> List[dict]:
        history, results = [], []
        for lang, text in messages:
            r = self.translate(text, lang, history, max_new_tokens)
            self.show(r)
            history.append(r["turn"])
            results.append(r)
        return results


def interactive_pipeline(pt: PipelineTranslator, max_new_tokens: int = 128):
    print('Type or paste messages starting with "en:" or "zh:" (several on one line are fine).')
    print('"reset" = new conversation, "quit" = stop (type it on its own line).')
    print(f"Each message: greedy translation + {pt.n_candidates} sampled candidates, best one picked by COMET.")
    history: List[Turn] = []
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
        messages = split_messages(line)
        if not messages:
            print('Start the line with a language code, e.g. "en: Hello" or "zh: 你好".')
            continue
        for lang, text in messages:
            r = pt.translate(text, lang, history, max_new_tokens)
            pt.show(r)
            history.append(r["turn"])
