"""Shared helpers for the context-aware chat MT pipeline.

This module factors out code that was previously duplicated across the
preprocessing notebooks (`notebooks/preprocess_data_wmt24_*.ipynb`,
`notebooks/make_few_shot_dev_instructions.ipynb`) and the vLLM scripts, so that
new language pairs (en<->zh) can be processed with *exactly* the same prompt
formats and data layout as the original ones.

Nothing in here changes the original methodology:
  * prompt strings are byte-identical to the ones produced by the notebooks
    (verified against the committed en-de instructions, see
    `scripts/make_instructions.py --verify`);
  * `HFBackend` is a drop-in replacement for the handful of vLLM features the
    scripts use (epsilon sampling with temperature + min_p, greedy decoding,
    prompt log-probs and generated-token log-probs) so the pipeline can run on
    machines without vLLM / GPUs. vLLM stays the default everywhere.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------

# Original languages (unchanged) + Chinese.
CODE_LANG_DICT = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "pt-br": "Brazilian Portuguese",
    "ko": "Korean",
    "nl": "Dutch",
    "zh": "Chinese",
}

# sacreBLEU tokenizer per target language (tower-eval `bleu: tokenizer:` arg).
# The original configs only override the tokenizer for Korean (ko-mecab);
# Chinese needs sacreBLEU's `zh` tokenizer, everything else uses the default.
BLEU_TOKENIZER = {"ko": "ko-mecab", "zh": "zh"}


def split_lp(lp: str):
    """'en-pt-br' -> ('en', 'pt-br'); 'zh-en' -> ('zh', 'en')."""
    if lp.startswith("pt-br-"):
        return "pt-br", lp[len("pt-br-") :]
    src, tgt = lp.split("-", 1)
    return src, tgt


# ---------------------------------------------------------------------------
# tower-eval I/O (with a faithful fallback when tower-eval is not installed)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on environment
    from tower_eval.utils import read_lines, write_lines  # noqa: F401
except Exception:  # tower-eval (and its vLLM dependency) is optional

    def read_lines(path, unescape_newline: bool = False) -> List[str]:
        """Same semantics as tower_eval.utils.read_lines."""
        with open(path, encoding="utf-8") as f:
            lines = [l[:-1] for l in f.readlines()]
        if unescape_newline:
            lines = [l.replace("\\n", "\n") for l in lines]
        return lines

    def write_lines(
        path,
        lines: Iterable[str],
        escape_newline: bool = False,
        escape_return_char: bool = True,
        verbose: bool = True,
    ) -> None:
        """Same semantics as tower_eval.utils.write_lines."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out_lines = []
        for line in lines:
            if escape_return_char:
                line = line.replace("\r", "\\r")
            if escape_newline:
                line = line.replace("\n", "\\n")
            out_lines.append(line)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines((f"{l}\n" for l in out_lines))


def load_jsonl(path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------------------
# Prompt formats (identical to the notebooks)
# ---------------------------------------------------------------------------
# Chat templates, as rendered by `tokenizer.apply_chat_template(...,
# add_generation_prompt=True)` for the Tower (ChatML) and Llama-3 tokenizers.
TEMPLATES = {
    # TowerInstruct-13B-v0.2 tokenizer, user turn only
    "chatml": "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    # TowerInstruct-v0.2-w-chat-mt-data tokenizer, empty system turn + user turn
    "chatml_empty_sys": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    # Tower-Llama3-70B, empty system turn
    "llama3_empty_sys": "<|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    # `*_no_template` prompts (used for gpt-4o)
    "no_template": "{content}",
}


def no_context_content(src: str, src_lang: str, tgt_lang: str) -> str:
    return f"Translate the following {src_lang} source text to {tgt_lang}:\n{src_lang}: {src}\n{tgt_lang}: "


def context_content(
    src: str, src_lang: str, tgt_lang: str, previous_sources: Sequence[str]
) -> str:
    initial_string = "Context: "
    for s in previous_sources:
        initial_string += f"{s}\n"
    final_string = (
        initial_string
        + f"\nTranslate the {src_lang} source text to {tgt_lang}, given the context."
    )
    return final_string + f"\n{src_lang}: {src}\n{tgt_lang}: "


def iter_conversations(records: List[dict]):
    """Yield lists of consecutive records sharing the same doc_id (same logic as
    the `register_convo` loop in the notebooks)."""
    convo: List[dict] = []
    for r in records:
        if convo and r["doc_id"] != convo[-1]["doc_id"]:
            yield convo
            convo = []
        convo.append(r)
    if convo:
        yield convo


def build_prompts(
    records: List[dict],
    template: str,
    use_context: bool,
    n_turns: Optional[int] = None,
    src_key: str = "src",
) -> List[str]:
    """Build one prompt per record of a *bilingual* conversation file (both
    directions interleaved, i.e. raw_data/mt/<dataset>.<xx>/test.jsonl).

    use_context=False reproduces `instructions/no_context*`,
    use_context=True reproduces `instructions/full_context*` (optionally
    truncated to the last `n_turns` utterances, `*_{n}_turns`).
    """
    tmpl = TEMPLATES[template]
    prompts = []
    for convo in iter_conversations(records):
        for i, row in enumerate(convo):
            src_lang = CODE_LANG_DICT[row["source_language"]]
            tgt_lang = CODE_LANG_DICT[row["target_language"]]
            if not use_context or i == 0:
                content = no_context_content(row[src_key], src_lang, tgt_lang)
            else:
                previous = [r[src_key] for r in convo[:i]]
                if n_turns is not None and len(previous) >= n_turns:
                    previous = previous[len(previous) - n_turns :]
                content = context_content(row[src_key], src_lang, tgt_lang, previous)
            prompts.append(tmpl.format(content=content))
    return prompts


def assistant_prefix_token_ids(tokenizer, template: str = "chatml") -> List[int]:
    """Token-id sequence that separates the prompt from the answer.

    The original PCXMI scripts hard-code this for the Tower tokenizers
    ([32005, 29871, 13, 32006, 20255, 13] / [32000, ...]). For any other
    tokenizer we derive it from the chat template so the same "find the last
    occurrence of the separator" gating can be used."""
    if template.startswith("chatml"):
        sep = "<|im_end|>\n<|im_start|>assistant\n"
    elif template.startswith("llama3"):
        sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        raise ValueError(f"No assistant separator for template {template}")
    return tokenizer.encode(sep, add_special_tokens=False)


def find_token_for_gating(lst, token_pattern):
    """Find the last occurrence of a token_pattern in a list (same as in
    scripts/pcxmi.py)."""
    token_pattern_len = len(token_pattern)
    for j in range(len(lst) - token_pattern_len, -1, -1):
        if lst[j : j + token_pattern_len] == token_pattern:
            return j
    raise ValueError("Token pattern not found in the list.")


def write_logprob_files(output_dir: Path, suffix: Optional[str], log_probs_list) -> None:
    """Write log_probs[_<suffix>].jsonl + mean/max/min/sum files (same layout as
    pcxmi/<context>/mt/<dataset>.<lp>/<model>/ and pcxmi_hyps/...)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sfx = f"_{suffix}" if suffix else ""
    with open(output_dir / f"log_probs{sfx}.jsonl", "w") as f_jsonl, open(
        output_dir / f"mean_log_probs{sfx}.txt", "w"
    ) as f_mean, open(output_dir / f"max_log_probs{sfx}.txt", "w") as f_max, open(
        output_dir / f"min_log_probs{sfx}.txt", "w"
    ) as f_min, open(
        output_dir / f"sum_log_probs{sfx}.txt", "w"
    ) as f_sum:
        for log_probs in log_probs_list:
            # coalesce to zero if no tokens (happens once in nl-en)
            if log_probs == []:
                log_probs = [-0.0]
            json.dump({"log_probs": log_probs}, f_jsonl)
            f_jsonl.write("\n")
            f_mean.write(f"{sum(log_probs) / len(log_probs)}\n")
            f_max.write(f"{max(log_probs)}\n")
            f_min.write(f"{min(log_probs)}\n")
            f_sum.write(f"{sum(log_probs)}\n")


# ---------------------------------------------------------------------------
# HuggingFace transformers backend (vLLM substitute for CPU / no-vLLM setups)
# ---------------------------------------------------------------------------
class HFBackend:
    """Minimal re-implementation of the vLLM calls used in this repository.

    - generate(prompts, temperature, min_p, max_tokens): temperature=0 ->
      greedy; otherwise ancestral sampling with temperature and min_p
      (epsilon sampling as in scripts/generate_candidates.py).
    - generate_with_logprobs(prompts, max_tokens): greedy decoding returning
      the log-prob of every generated token (vLLM `logprobs=1`), including EOS
      as vLLM does.
    - prompt_logprobs(token_id_lists): log-prob of each prompt token given its
      prefix (vLLM `prompt_logprobs=1`); first position is None.
    """

    def __init__(self, model_name_or_path: str, device: Optional[str] = None, seed: int = 42, dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        torch.manual_seed(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if dtype is None:
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
        self.model.to(self.device).eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_tokenizer(self):
        return self.tokenizer

    def _encode(self, prompt: str) -> List[int]:
        # vLLM tokenizes the raw prompt with the tokenizer defaults
        return self.tokenizer.encode(prompt)

    def _sample_one(self, input_ids: List[int], temperature: float, min_p: float, max_tokens: int, want_logprobs: bool):
        torch = self.torch
        eos = self.tokenizer.eos_token_id
        ids = torch.tensor([input_ids], device=self.device)
        past = None
        out_ids, out_lps = [], []
        with torch.no_grad():
            for _ in range(max_tokens):
                res = self.model(input_ids=ids, past_key_values=past, use_cache=True)
                past = res.past_key_values
                logits = res.logits[0, -1].float()
                logprobs = torch.log_softmax(logits, dim=-1)
                if temperature == 0.0:
                    nxt = int(torch.argmax(logits))
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    if min_p and min_p > 0:
                        probs = torch.where(probs < min_p * probs.max(), torch.zeros_like(probs), probs)
                        probs = probs / probs.sum()
                    nxt = int(torch.multinomial(probs, 1))
                out_ids.append(nxt)
                out_lps.append(float(logprobs[nxt]))
                if nxt == eos:
                    break
                ids = torch.tensor([[nxt]], device=self.device)
        text_ids = out_ids[:-1] if out_ids and out_ids[-1] == eos else out_ids
        text = self.tokenizer.decode(text_ids, skip_special_tokens=True)
        return (text, out_lps) if want_logprobs else text

    def generate(self, prompts: List[str], temperature: float = 0.0, min_p: float = 0.0, max_tokens: int = 1024, use_tqdm: bool = True) -> List[str]:
        from tqdm import tqdm

        it = tqdm(prompts, desc="generate") if use_tqdm else prompts
        return [self._sample_one(self._encode(p), temperature, min_p, max_tokens, False) for p in it]

    def generate_with_logprobs(self, prompts: List[str], max_tokens: int = 1024, use_tqdm: bool = True):
        from tqdm import tqdm

        it = tqdm(prompts, desc="generate+logprobs") if use_tqdm else prompts
        return [self._sample_one(self._encode(p), 0.0, 0.0, max_tokens, True) for p in it]

    def prompt_logprobs(self, token_id_lists: List[List[int]], use_tqdm: bool = True) -> List[List[Optional[float]]]:
        torch = self.torch
        from tqdm import tqdm

        out = []
        it = tqdm(token_id_lists, desc="prompt logprobs") if use_tqdm else token_id_lists
        with torch.no_grad():
            for toks in it:
                ids = torch.tensor([toks], device=self.device)
                logits = self.model(input_ids=ids).logits[0].float()
                lps = torch.log_softmax(logits[:-1], dim=-1)
                tgt = ids[0, 1:]
                token_lps = lps.gather(1, tgt.unsqueeze(1)).squeeze(1).tolist()
                out.append([None] + token_lps)
        return out


def repo_root() -> Path:
    return Path(os.environ.get("CHAT_MT_ROOT", Path(__file__).resolve().parent.parent))
