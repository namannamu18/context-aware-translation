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

TowerInstruct-7B-v0.2 officially covers Chinese. The individual scripts accept the same arguments as before
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

### Paid components and free alternatives

Paid components (GPT-4o translation baseline, GPT-4 GEMBA/ContextMQM) are not needed for the pipeline.

| Paid component | Free replacement provided | Status |
|---|---|---|
| `gpt-4o` baseline (`configs/*_openai.yaml`) | `configs/zh/{no,full}_context_open_llm.yaml` with `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0, ChatML — the Tower prompts are used unchanged) | config provided; not run here (no HF access) |
| GPT-4 GEMBA / ContextMQM (`run_context_llm.py`) | any open instruct model behind an OpenAI-compatible server (`vllm serve Qwen/Qwen2.5-72B-Instruct`, or `scripts/serve_openai_compatible.py`) via `--base_url --judge_model` | code path tested end-to-end on en↔zh with a local server; judge quality of a real open model not measured |

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
