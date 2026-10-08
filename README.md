# A Context-aware Framework for Translation-mediated Conversations

[Paper (arXiv)](https://arxiv.org/abs/2412.04205) | [WMT 2024 Chat Shared Task Submission](https://aclanthology.org/2024.wmt-1.100.pdf)

## Installation

Install [`tower-eval`](https://github.com/deep-spin/tower-eval) to run generation and evaluation configs.

## Pipeline

Data prep → Candidate generation (epsilon sampling) → Decoding (greedy / MBR / contrastive) → PCXMI analysis → Evaluation

## Languages & Models

**Language pairs** — WMT24 Chat: en↔{de, fr, pt-br, nl, ko}; BConTrasT: en↔de; BMELD (+ a small synthetic test set): en↔zh (see [English ↔ Chinese](#english--chinese-en-zh--zh-en))

| Model | Role |
|---|---|
| `Unbabel/TowerInstruct-7B-v0.2` | Base 7B; candidate generation, PCXMI |
| `TowerInstruct-7B-w-chat` / `TowerChat-7B` | Fine-tuned 7B on chat MT data |
| `Tower-Llama3-70B` | Large-scale 70B baseline |
| `gpt-4o` | Proprietary baseline (no-template) |

## Evaluation Metrics

- **Translation quality**: COMET-22, ChrF, BLEU, COMET-Kiwi, MetricX-XXL, XCOMET-XXL
- **Discourse**: MuDA (lexical cohesion, formality, verb form, pronouns)
- **LLM-as-judge**: GEMBA-MQM via GPT-4 (context-aware, 1-shot)
- **Context sensitivity**: PCXMI (log-prob difference between full-context and no-context prompts)

## Repository Structure

### Data

| Folder | Description |
|---|---|
| `downloaded_data/` | Raw CSVs from WMT24 Chat Task, BConTrasT, and MAIA |
| `raw_data/` | Processed JSONL per dataset and language pair |
| `instructions/` | Pre-formatted LLM prompts (per experimental condition) used by `tower-eval` |

### Configs & Generation

| Folder | Description |
|---|---|
| `configs/` | `tower-eval` YAML configs (~20 conditions: context windows, prompt formats, model scales, few-shot) |
| `generations/` | Model translation outputs, one subfolder per condition |
| `candidates/` | 100 epsilon-sampled candidates per segment (for MBR) |

### Decoding & Analysis

| Folder | Description |
|---|---|
| `mbr_outputs/` | MBR-selected translations (dev/test/train) using COMET and context-COMET variants |
| `contrast_decode/` | Contrastive decoding outputs (interpolating context vs. no-context logits) |
| `pcxmi/` | Per-token log-prob scores for TowerInstruct under ~14 context settings |
| `pcxmi_hyps/` | Cross-condition PCXMI (full-context prompts on no-context outputs, and vice versa) |
| `alti_analysis/` | ALTI attention analysis outputs |

### Results & Evaluation

| Folder | Description |
|---|---|
| `evaluations/` | Automatic metric scores per condition (COMET, ChrF, BLEU, etc.) |
| `paper_results/` | Consolidated CSVs with all outputs + scores per LP, plus MuDA JSONs |
| `tacl_results/` | Extended result CSVs for the TACL submission (includes GEMBA, turn-level breakdowns) |
| `plots/` | ~50 PDF figures included in the paper |
| `submission_unbabel+it/` | Official WMT24 shared task submissions (primary + 2 contrastive per LP) |

### Code

| Folder | Description |
|---|---|
| `scripts/` | Candidate generation, MBR decoding, contrastive decoding, PCXMI computation, MuDA evaluation, GEMBA-MQM scoring, fine-tuning data prep |
| `notebooks/` | Notebooks for data preprocessing, evaluation, analysis, plotting, and statistical significance testing |

## English ↔ Chinese (en-zh / zh-en)

The repository also supports **English→Chinese and Chinese→English** with the same methodology
(context vs. no-context prompts, epsilon-sampling candidates, COMET / context-COMET MBR, contrastive
decoding, P-CXMI, automatic evaluation). All original language pairs, data, configs and outputs are untouched.

### Data

| Dataset | Type | Segments (en→zh / zh→en) | Conversations |
|---|---|---|---|
| `bmeld_train` | **real** — BMELD ([Liang et al., ACL 2021](https://aclanthology.org/2021.acl-long.444/), [CPCC repo](https://github.com/XL2248/CPCC)) | 5585 / 4402 | 1036 |
| `bmeld_dev` | **real** — BMELD | 567 / 517 | 108 |
| `bmeld_test` | **real** — BMELD | 1466 / 1135 | 274 |
| `synthetic_chat_test` | **SYNTHETIC** customer-support chats, written for pipeline testing only | 34 / 25 | 4 |

BMELD is a bilingual chat translation corpus built on MELD (Friends dialogues) with human post-edited
Chinese translations. Following BMELD/BConTrasT (and `preprocess_ench.py` in CPCC), utterances by Ross,
Joey and Rachel are Chinese-speaker turns (zh→en, `sender=customer`) and all others are English-speaker
turns (en→zh, `sender=agent`), mirroring the agent(English)/customer(xx) roles of WMT24 Chat. The only
text normalisation is removing the space MELD puts before punctuation (`"it ."` → `"it."`).

`synthetic_chat_test` is clearly labelled (`doc_id` starts with `SYNTHETIC-`, `client_id=SYNTHETIC`,
`tags=['synthetic']`); its references were hand-written and not validated by professional translators —
**do not report results on it**. There is no WMT24 Chat data for Chinese.

Files follow exactly the layout of the WMT24 data:
`downloaded_data/<data>_<split>.en-zh.csv`, `raw_data/mt/<data>_<split>.{zh,en-zh,zh-en}/test.jsonl`,
`instructions/<condition>/mt/<data>_<split>.{en-zh,zh-en}/instructions.txt`, and
`paper_results_zh/<data>/<split>.en-zh.csv` (input of MBR / contrastive decoding).

```bash
python scripts/prepare_zh_data.py              # downloads BMELD + writes the synthetic set
python scripts/make_instructions.py --verify   # regenerates committed en-de/en-ko/... prompts: byte-identical
python scripts/make_instructions.py --datasets bmeld_train --conditions no_context full_context
python scripts/make_instructions.py            # all prompt conditions for bmeld_test/dev + synthetic_chat_test
```

`scripts/make_instructions.py` factors out the prompt code of the preprocessing notebooks
(`scripts/chat_mt_utils.py`); `--verify` checks that it reproduces the committed instructions of the
existing language pairs byte-for-byte. Conditions: `no_context`, `full_context`, `*_empty_sys`,
`*_{2,6,10,15,20}_turns`, `*_empty_sys_llama3`, `*_no_template`, `*_5_shot` (examples from `bmeld_train`).

### Running the pipeline

With GPUs (as in the paper): `tower-eval gen-eval --config configs/zh/<condition>.yaml` (configs mirror the
original ones; BLEU uses sacreBLEU's `zh` tokenizer for en→zh), then

```bash
# everything else (candidates -> MBR -> contrastive decoding -> P-CXMI -> report) in one go:
ROOT=. MODEL=Unbabel/TowerInstruct-7B-v0.2 MODEL_NAME=TowerInstruct-7B-v0.2 BACKEND=vllm \
  N_CANDIDATES=100 CONTEXT_SIZES="0 2 6 10 15" bash scripts/run_zh_pipeline.sh
```

TowerInstruct-7B-v0.2 officially covers Chinese. No GPU? See [`kaggle/README.md`](kaggle/README.md) to run everything on Kaggle's free T4 GPUs. The individual scripts accept the same arguments as before
plus new, optional ones (defaults reproduce the original runs):

| Script | New options |
|---|---|
| `generate_candidates.py` | `--root_dir --model --datasets --lps --prompts --n_candidates --backend {vllm,hf}` |
| `pcxmi.py` | `--datasets --lps --targets {ref,hyp,src} --ref_source {paper_results,raw_data} --sep_tokens {tower,chatml,llama3} --backend` (the `hyp`/`src` variants were commented out before; a stray `breakpoint()` was removed) |
| `pcxmi_hyps.py` | `--datasets --lps --model --full_context_prompt_name --no_context_prompt_name --sep_tokens --backend --gen_backend` |
| `run_context_comet_mbr.py` | `--data_dir --num_candidates --gen_backend --comet_model (HF id or .ckpt) --utility {comet,chrf}`; runs on CPU if no GPU |
| `run_contrastive_decoding.py` | `--data_name --context_prompt --no_context_prompt --comet_model --max_new_tokens`; CPU fallback, tokenizer-agnostic padding |
| `run_context_llm.py` (GEMBA) | `zh` language, `--base_url --judge_model --data_dir --max_workers` (any OpenAI-compatible endpoint) |

New helper scripts: `generate_translations.py` (tower-eval-style greedy generation, vLLM or HF backend),
`evaluate_translations.py` (tower-eval-compatible `evaluation.json`), `tower_eval_offline_evaluate.py`
(tower-eval's own evaluator without vLLM), `pcxmi_summary.py`, `zh_pipeline_report.py`,
`check_zh_pipeline_outputs.py` (consistency checks), `subset_zh_dataset.py`, `smoke_models.py`,
`serve_openai_compatible.py`, `training_tower_chat/make_zh_chat_mt_data.py` (fine-tuning data, with/without
context, optionally MBR-distilled). MuDA supports Chinese: `get_muda_accuracy.py --tgt-lang zh`.
`translate_chat.py` translates your own chat messages (interactive) or a few BMELD conversations (`--quick_check`).
`translate_chat_pipeline.py` does the same with the full method (greedy, epsilon sampling, COMET MBR incl. the context-aware
variant) for typed conversations; it uses the same prompts and the same `run_mbr` as the batch pipeline. Typed messages have no
reference, so it prints translations, not scores.

**Repetition loops.** Greedy decoding occasionally gets stuck (e.g. `啊，啊，啊，…` until `max_tokens`).
`generate_translations.py --loop_retry_penalty 1.1` (`LOOP_RETRY_PENALTY=1.1` in `run_zh_pipeline.sh`; on by default
in the Kaggle notebook and in `translate_chat.py`) re-translates *only* such outputs (a short unit repeated ≥10 times
and far more often than in the source) with repetition penalty 1.1. All other outputs are byte-identical to plain
greedy decoding. A global `--repetition_penalty` also exists but changes normal translations too, so it is not recommended.
Default: off (paper setup).

### Fine-tuned system, judge and one-cell translator

The paper's main system is TowerInstruct-7B **fine-tuned on chat MT data with the conversation as context**
(`Unbabel/TowerInstruct-WMT24-Chat-7B`); its sampled candidates are re-ranked with context-aware COMET MBR, and that
"contextual MBR re-ranking" (`mbr-source` in `paper_results`) was the primary WMT24 submission
(`submission_unbabel+it/README.md`, `notebooks/test-eval.ipynb`). The same workflow for English↔Chinese, sized for Kaggle's free GPUs:

| Paper | Here |
|---|---|
| full fine-tuning (axolotl) on the WMT24 chat training data, with context | `scripts/finetune_lora.py`: **LoRA on a 4-bit base (QLoRA)** on BMELD-train, same data format (`full_context` instruction in the empty-system chat template, loss on the reference only), runtime bounded by `--time_budget_min`; `scripts/merge_lora.py` merges the adapter into a standalone checkpoint |
| 100 sampled candidates per message | 6 (`N_CANDIDATES`) |
| WMT24 chat dev/test sets | a subset of BMELD-test (`MAX_DOCS` conversations) |
| prompts of the fine-tuned model (`*_empty_sys`) | `PROMPT_SUFFIX=_empty_sys` in `scripts/run_zh_pipeline.sh` (also `STAGES`, `CD_VARIANTS`, `MBR_STYLES`) |
| `paper_results/{dev,test}.<lp>.csv`: one row per message, a column per system | `scripts/consolidate_zh_results.py` → `results_zh/<data>/<split>.en-zh.csv` (+ `.summary.md`): base / fine-tuned greedy with and without context, `mbr`, `mbr-source` (primary), `mbr-hyp`, contrastive decoding, COMET per message |
| GPT-4 GEMBA-MQM judge, 1-shot examples from human annotations | `run_context_llm.py --provider gemini` (Gemini through Google AI Studio's free tier; key from `GEMINI_API_KEY`, never from the command line) with an en-zh 1-shot example **written for this repository**. Scores are **not comparable** to the paper's GPT-4 numbers. Note on the inherited code: with `--context_mode target` (default, as in the original `run_context_llm.py`) the judge's context shows the *translations* of the earlier messages labelled with the source language (e.g. `Agent (English): 你好…`), whereas the prompt's own 1-shot examples show the original messages; `--context_mode source` gives the latter. Robust to the answer formats of other LLMs (an unparsable answer is NaN, never a perfect score), retries, rate limit, resumable |
| files only (no interactive use) | `scripts/translate_chat_pipeline.py`: type a conversation, every message is translated with the earlier ones as context by the fine-tuned model (6 candidates, context-aware COMET MBR = the paper's primary system). **Optionally the judge grades each translation right after it was produced** (`scripts/judge_chat.py`, same prompt and parser as `run_context_llm.py`; it only grades, it never changes the translation) |

Kaggle notebooks (details in [`kaggle/README.md`](kaggle/README.md)): `train_eval_zh_kaggle.ipynb` (one time: fine-tune + evaluate),
`judge_zh_kaggle.ipynb` (CPU only), `translate_zh_kaggle.ipynb` (your own conversations, one cell, no training or evaluation again).
The whole workflow was tested on CPU with the tiny test models (`scripts/run_zh_finetune_smoke_test.sh`, results in
`zh_smoke_tests/run4_finetuned/`); it has **not** been run on a GPU with the real Tower and COMET models yet, so the first Kaggle run may need small fixes.
Deviations from the paper to keep in mind: LoRA instead of full fine-tuning, BMELD (TV dialogue) instead of customer-support chats, 6 instead of 100 candidates,
a small test subset, Gemini instead of GPT-4. Numbers from this setup are a sanity check of the pipeline, not research results.

### Paid components and free alternatives

Paid components (GPT-4o translation baseline, GPT-4 GEMBA/ContextMQM) are not needed for the pipeline.

| Paid component | Free replacement provided | Status |
|---|---|---|
| `gpt-4o` baseline (`configs/*_openai.yaml`) | `configs/zh/{no,full}_context_open_llm.yaml` with `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0, ChatML — the Tower prompts are used unchanged) | config provided; not run here (no HF access) |
| GPT-4 GEMBA / ContextMQM (`run_context_llm.py`) | any open instruct model behind an OpenAI-compatible server (`vllm serve Qwen/Qwen2.5-72B-Instruct`, or `scripts/serve_openai_compatible.py`) via `--base_url --judge_model` | code path tested end-to-end on en↔zh with a local server; judge quality of a real open model not measured |
| GPT-4 GEMBA-MQM (context-aware) as a judge | **Gemini via Google AI Studio's free tier**: `run_context_llm.py --provider gemini`, `kaggle/judge_zh_kaggle.ipynb` (a ChatGPT Pro subscription does *not* include OpenAI API access) | tested against a fake API server (retries, formats, resume); not run with a real key; free-tier limits apply |

COMET-22 (`Unbabel/wmt22-comet-da`), used for MBR and evaluation, is already free/open.
`--utility chrf` is only a fallback for environments without any COMET checkpoint; the paper's method is COMET MBR.

### CPU smoke tests (no model downloads)

`scripts/smoke_models.py` trains a tiny Llama-architecture LM (5.5M parameters, Tower-style ChatML special
tokens) on BMELD train prompts and builds a randomly initialised COMET checkpoint, so every stage can be
executed offline:

```bash
python scripts/smoke_models.py lm --out smoke_models/tiny-chatml-lm --minutes 20
python scripts/smoke_models.py comet --out smoke_models/tiny-comet
PYTHON=python ROOT=runs/smoke1 DATA_NAME=bmeld SPLIT=test MAX_DOCS=6 BACKEND=hf SEP_TOKENS=chatml \
  MODEL=smoke_models/tiny-chatml-lm MODEL_NAME=tiny-chatml-lm N_CANDIDATES=4 MAX_TOKENS=48 \
  COMET_MODEL=smoke_models/tiny-comet/checkpoints/model.ckpt CONTEXT_SIZES=2 bash scripts/run_zh_pipeline.sh
python scripts/check_zh_pipeline_outputs.py --root_dir runs/smoke1 --model smoke_models/tiny-chatml-lm \
  --model_name tiny-chatml-lm --n_candidates 4
```

These models only test the plumbing — **their scores are meaningless and are not research results.**
See `zh_smoke_tests/README.md` for the runs performed while adding en↔zh support.

## Citation

```bibtex
@article{pombal2024context,
  title={A context-aware framework for translation-mediated conversations},
  author={Pombal, Jos{\'e} and Agrawal, Sweta and Fernandes, Patrick and Zaranis, Emmanouil and Martins, Andr{\'e} FT},
  journal={arXiv preprint arXiv:2412.04205},
  year={2024}
}
```

```bibtex
@inproceedings{pombal2024improving,
  title={Improving context usage for translating bilingual customer support chat with large language models},
  author={Pombal, Jos{\'e} and Agrawal, Sweta and Martins, Andr{\'e} FT},
  booktitle={Proceedings of the Ninth Conference on Machine Translation},
  pages={993--1003},
  year={2024}
}
```
