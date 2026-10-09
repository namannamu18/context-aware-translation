"""Judge ONE translation right after it was produced: the paper's context-aware GEMBA-MQM judge (scripts/run_context_llm.py), here
with Gemini's free API tier instead of GPT-4. The judge only grades; it never changes the translation (the choice among the sampled
candidates is made by COMET before, see scripts/translate_chat_pipeline.py).

Prompt, answer parser and scoring are those of run_context_llm.py: score = -(10 x critical + 5 x major + 1 x minor errors), closer to 0 is
better; an unparsable answer gives no score (never a perfect one). The key is read from the environment (Kaggle: Add-ons -> Secrets ->
GEMINI_API_KEY) and never printed.

    judge = ChatJudge.create()            # None (with a message) if there is no key
    judge.score(history, "en", "Hello", "你好")
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_context_llm as rcl  # noqa: E402
from llm_fewshot_examples import TEMPLATE_GEMBA_CONTEXT_MQM_1shot  # noqa: E402

LANG = {"en": "English", "zh": "Chinese"}
SENDER = {"en": "Agent", "zh": "Customer"}   # the English speaker is the agent, as in the BMELD data of this repository


class ChatJudge:
    def __init__(self, provider: str = "gemini", judge_model: str = "gemini-2.5-flash", base_url: Optional[str] = None,
                 api_key_env: Optional[str] = None, rpm: float = 10, context_size: int = 8, context_mode: str = "target",
                 max_tokens: Optional[int] = None, extra_body: Optional[dict] = None, retries: int = 3):
        assert context_mode in ("target", "source")
        args = SimpleNamespace(provider=provider, base_url=base_url, judge_model=judge_model, api_key_env=api_key_env)
        self.client, self.minimal = rcl.make_client(args)       # SystemExit without a key
        try:   # get_response retries on its own (short); the client's hidden retries could stall the conversation for a long time
            self.client = self.client.with_options(max_retries=0, timeout=60)
        except Exception:
            pass
        self.judge_model = args.judge_model
        self.max_tokens = max_tokens or (1024 if self.minimal else 100)
        self.extra_body = extra_body
        self.limiter = rcl.RateLimiter(rpm)
        self.context_size = context_size
        self.context_mode = context_mode
        self.retries = retries
        self.enabled = True
        self.template = TEMPLATE_GEMBA_CONTEXT_MQM_1shot("enzh_conversation")

    @classmethod
    def create(cls, **kwargs) -> Optional["ChatJudge"]:
        """The judge, or None (with a message) when no API key is available: the translator then works without it."""
        try:
            return cls(**kwargs)
        except (SystemExit, KeyError) as e:
            print(f"(judge disabled: {e if str(e) else 'no API key'})")
            return None

    def prompt(self, history: list, lang: str, text: str, translation: str) -> list:
        """The messages sent to the judge. `history` = previous Turns (lang, text, comet_best, final)."""
        tgt = "zh" if lang == "en" else "en"
        window = history[max(0, len(history) - self.context_size):] if self.context_size > 0 else []
        lines = []
        for t in window:
            shown = (t.final or t.text) if self.context_mode == "target" else t.text
            lines.append(f"{SENDER[t.lang]} ({LANG[t.lang]}): {shown}")
        data = {"context": "\n".join(lines), "sender": SENDER[lang], "source_lang": LANG[lang], "source_seg": text,
                "target_lang": LANG[tgt], "target_seg": translation}
        return rcl.apply_template(self.template, data)

    def score(self, history: list, lang: str, text: str, translation: str) -> dict:
        """{"score", "critical", "major", "minor", "answer"}; score is None if the judge could not be reached or its answer was unreadable."""
        answer = rcl.get_response(self.client, self.prompt(history, lang, text, translation), self.judge_model, self.max_tokens,
                                  self.minimal, self.extra_body, self.limiter, None, self.retries)
        if answer is None:
            return {"score": None, "error": "no answer from the judge (rate limit, network or model name?)", "answer": None}
        s = rcl.parse_mqm_answer(answer)
        if np.isnan(s.iloc[0]):
            return {"score": None, "error": "unreadable answer", "answer": answer}
        return {"score": int(s.iloc[0]), "critical": int(s.iloc[1]), "major": int(s.iloc[2]), "minor": int(s.iloc[3]), "answer": answer}


def describe(j: Optional[dict], model: str = "") -> str:
    """One compact line for the screen."""
    if j is None:
        return ""
    if j.get("score") is None:
        return f"   judge: no score ({j.get('error')})"
    errors = [l.strip() for l in (j.get("answer") or "").splitlines() if l.strip() and "no-error" not in l.lower() and not l.strip().endswith(":")]
    counts = f"{j['critical']} critical, {j['major']} major, {j['minor']} minor"
    return f"   judge{' (' + model + ')' if model else ''}: MQM {j['score']} ({counts})" + ("".join(f"\n      {e}" for e in errors[:6]))
