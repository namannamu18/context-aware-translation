# en↔zh pipeline smoke tests (plumbing verification — NOT research results)

> **Read this first.** HuggingFace Hub (and other model hosts) were blocked by the network policy of the
> environment in which en↔zh support was added, so no pretrained model (TowerInstruct, COMET, Qwen, …) could
> be downloaded. To still execute every stage of the pipeline on real en↔zh data, two *smoke-test models*
> were built locally with `scripts/smoke_models.py`:
>
> * `tiny-chatml-lm`: 5.5M-parameter Llama-architecture LM with Tower-style ChatML special tokens, trained
>   from scratch for ~55 CPU-minutes on BMELD **train** prompts (with and without context). It outputs
>   Chinese for en→zh and English for zh→en, but its translations are poor.
> * `tiny-comet`: a **randomly initialised** COMET RegressionMetric (avg pooling, so context-COMET works).
>
> **All numbers below are meaningless as MT quality / context-usage measurements.** They only show that each
> stage ran, produced aligned outputs, and that the scores are computed consistently. They must not be
> compared with the paper or reported anywhere.

## What was run

`scripts/run_zh_smoke_tests.sh` → three runs of `scripts/run_zh_pipeline.sh` on **different test data**, each
followed by `scripts/check_zh_pipeline_outputs.py` and by tower-eval's own evaluator on `configs/zh/full_context.yaml`
(`scripts/tower_eval_offline_evaluate.py`):

```bash
python scripts/smoke_models.py lm --out smoke_models/tiny-chatml-lm --minutes 20
python scripts/smoke_models.py lm --init_from smoke_models/tiny-chatml-lm --out smoke_models/tiny-chatml-lm2 --minutes 35 --lr 7e-4
python scripts/smoke_models.py comet --out smoke_models/tiny-comet
OUT=runs LM=smoke_models/tiny-chatml-lm2 COMET_CKPT=smoke_models/tiny-comet/checkpoints/model.ckpt \
  N_CANDIDATES=4 MAX_TOKENS=48 bash scripts/run_zh_smoke_tests.sh
```

Each run executes, for **both en→zh and zh→en**: dataset subset → prompts (`no_context`, `full_context`,
`*_empty_sys`, `full_context_6_turns`) → greedy decoding (no/full context) → evaluation (chrF, BLEU with the
sacreBLEU `zh` tokenizer for Chinese targets, COMET) → epsilon sampling (T=0.7, min_p=0.02, 4 candidates instead of 100)
→ MBR for both candidate pools: source-only COMET MBR, context-COMET MBR with source context and with
COMET-best-translation context, windows 0 and 2 → contrastive decoding (weights c1/nc0, c0/nc1, c1/nc1, c5/nc1,
plus sampled c1/nc1 ×4) → P-CXMI on reference / hypothesis / source + cross-condition P-CXMI (`pcxmi_hyps`) → report.

Checks per run (`checks.txt`, 55 per run): prompt/source alignment and language names; full-context prompts contain
the whole conversation history of both speakers; line counts of generations, candidates, contrastive outputs;
every MBR selection is one of the candidates of the same segment; **P-CXMI gating**: #ref log-probs = #reference
tokens + EOS and #src log-probs = #source tokens for every segment in both context settings; evaluation files have
one segment score per segment; language sanity (share of CJK characters).

## Results

| Run | Data | Pipeline | Checks | tower-eval vs. `evaluate_translations.py` |
|---|---|---|---|---|
| run1 | `bmeld_test` conversations 1–8 (30 en→zh + 33 zh→en segments) | exit 0 | 55/55 | identical (all segments, chrF + BLEU) |
| run2 | `bmeld_dev` conversations 21–28 (50 + 35) | exit 0 (first attempt failed in the report step, see below) | 55/55 | identical |
| run3 | `synthetic_chat_test`, all 4 conversations (34 + 25) | exit 0 | 55/55 | identical |

Language sanity: CJK-character share of en→zh outputs 0.80–0.86 (references 0.83–0.86); zh→en outputs 0.00–0.002.

Issues found and fixed while running:

* run2 (first attempt): `zh_pipeline_report.py` crashed on the empty `output-select-comet` column that
  `run_context_comet_mbr.py` writes on the dev split when COMET is unavailable (`first_attempt_error.txt`).
  Fixed (numeric coercion) and run2 was re-run from scratch → exit 0.
* tower-eval evaluation of en→zh first gave BLEU 0.0 while `evaluate_translations.py` gave 4.40: the offline helper
  did not flatten the subtask `eval_args:` block the way `tower-eval gen-eval` does, so the `zh` BLEU tokenizer was
  ignored. Fixed in `scripts/tower_eval_offline_evaluate.py`; afterwards both implementations agree exactly.
* Found earlier with a random-weight debug model: contrastive decoding ran until `max_length` (≈150 s/segment on
  CPU) → added an optional `--max_new_tokens`; MBR selections are whitespace-stripped by the original `read_file()`
  (not a bug — the check compares stripped strings).

In run2 the MBR output files keep the runner's standard names (`comet_eps*`) although the utility was chrF.

### run1: bmeld_test, conversations 1–8, COMET MBR (smoke COMET checkpoint)

Consistency checks: **55/55 checks passed**. tower-eval (`configs/zh/full_context.yaml`) re-computed chrF/BLEU identical to `evaluate_translations.py` on every segment.

| Stage | en→zh chrF / BLEU | zh→en chrF / BLEU |
|---|---|---|
| greedy, no_context | 6.00 / 4.48 | 15.19 / 2.34 |
| greedy, full_context | 6.07 / 4.40 | 16.67 / 3.00 |
| MBR (full_context candidates) comet_eps | 5.95 / 4.39 | 15.66 / 2.03 |
| MBR (full_context candidates) comet_eps_context_comet-best_w2 | 5.70 / 4.37 | 15.42 / 2.00 |
| MBR (full_context candidates) comet_eps_context_source_w2 | 5.92 / 4.46 | 15.22 / 1.87 |
| MBR (no_context candidates) comet_eps | 6.50 / 4.53 | 14.42 / 1.51 |
| MBR (no_context candidates) comet_eps_context_comet-best_w2 | 6.63 / 4.78 | 13.22 / 1.37 |
| MBR (no_context candidates) comet_eps_context_source_w2 | 6.50 / 4.57 | 13.75 / 1.37 |
| contrastive decoding c0_nc1 | 6.01 / 4.49 | 15.11 / 2.28 |
| contrastive decoding c1_nc0 | 6.08 / 4.41 | 16.47 / 3.01 |
| contrastive decoding c1_nc1 | 6.10 / 4.50 | 16.81 / 2.98 |
| contrastive decoding c5_nc1 | 6.19 / 4.48 | 16.68 / 2.69 |

| P-CXMI (mean, full − no context) | en→zh | zh→en |
|---|---|---|
| pcxmi_ref | -0.040 | +0.058 |
| pcxmi_hyp | +0.017 | +0.032 |
| pcxmi_src | -0.057 | -0.195 |
| pcxmi_on_full_context_output | +0.282 | +0.259 |
| pcxmi_on_no_context_output | -0.101 | -0.238 |

Segments: en→zh 30, zh→en 33.

### run2: bmeld_dev, conversations 21–28, chrF-utility MBR; real `Unbabel/wmt22-comet-da` id → unreachable → COMET skipped (fallback path)

Consistency checks: **55/55 checks passed**. tower-eval (`configs/zh/full_context.yaml`) re-computed chrF/BLEU identical to `evaluate_translations.py` on every segment.

| Stage | en→zh chrF / BLEU | zh→en chrF / BLEU |
|---|---|---|
| greedy, no_context | 9.79 / 9.67 | 15.82 / 4.57 |
| greedy, full_context | 9.93 / 10.04 | 15.73 / 5.08 |
| MBR (full_context candidates) comet_eps | 9.51 / 9.16 | 16.16 / 4.75 |
| MBR (full_context candidates) comet_eps_context_comet-best_w2 | 9.17 / 8.64 | 16.10 / 3.98 |
| MBR (full_context candidates) comet_eps_context_source_w2 | 9.17 / 8.64 | 16.10 / 3.98 |
| MBR (no_context candidates) comet_eps | 9.61 / 8.91 | 18.22 / 3.30 |
| MBR (no_context candidates) comet_eps_context_comet-best_w2 | 9.65 / 8.70 | 18.50 / 3.99 |
| MBR (no_context candidates) comet_eps_context_source_w2 | 9.62 / 8.72 | 18.50 / 3.99 |
| contrastive decoding c0_nc1 | 9.74 / 9.63 | 15.87 / 4.86 |
| contrastive decoding c1_nc0 | 10.05 / 10.13 | 15.82 / 4.32 |
| contrastive decoding c1_nc1 | 10.75 / 11.11 | 16.59 / 4.83 |
| contrastive decoding c5_nc1 | 10.27 / 10.30 | 15.59 / 4.12 |

| P-CXMI (mean, full − no context) | en→zh | zh→en |
|---|---|---|
| pcxmi_ref | -0.044 | +0.044 |
| pcxmi_hyp | -0.002 | +0.014 |
| pcxmi_src | +0.004 | -0.024 |
| pcxmi_on_full_context_output | +0.347 | +0.290 |
| pcxmi_on_no_context_output | -0.209 | -0.268 |

Segments: en→zh 50, zh→en 35.

### run3: synthetic_chat_test (all 4 conversations), COMET MBR (smoke COMET checkpoint)

Consistency checks: **55/55 checks passed**. tower-eval (`configs/zh/full_context.yaml`) re-computed chrF/BLEU identical to `evaluate_translations.py` on every segment.

| Stage | en→zh chrF / BLEU | zh→en chrF / BLEU |
|---|---|---|
| greedy, no_context | 4.29 / 2.25 | 11.74 / 1.21 |
| greedy, full_context | 3.59 / 1.34 | 12.26 / 1.10 |
| MBR (full_context candidates) comet_eps | 3.21 / 1.09 | 12.57 / 1.21 |
| MBR (full_context candidates) comet_eps_context_comet-best_w2 | 3.17 / 1.14 | 12.80 / 1.23 |
| MBR (full_context candidates) comet_eps_context_source_w2 | 2.92 / 1.04 | 12.20 / 1.28 |
| MBR (no_context candidates) comet_eps | 3.66 / 1.42 | 11.59 / 0.97 |
| MBR (no_context candidates) comet_eps_context_comet-best_w2 | 3.53 / 1.39 | 12.36 / 0.91 |
| MBR (no_context candidates) comet_eps_context_source_w2 | 3.64 / 1.42 | 11.87 / 1.04 |
| contrastive decoding c0_nc1 | 3.88 / 1.60 | 12.01 / 1.17 |
| contrastive decoding c1_nc0 | 3.69 / 1.37 | 12.42 / 1.30 |
| contrastive decoding c1_nc1 | 3.94 / 1.51 | 12.27 / 1.27 |
| contrastive decoding c5_nc1 | 3.79 / 1.51 | 12.40 / 1.19 |

| P-CXMI (mean, full − no context) | en→zh | zh→en |
|---|---|---|
| pcxmi_ref | -0.016 | -0.037 |
| pcxmi_hyp | -0.015 | -0.130 |
| pcxmi_src | +0.042 | +0.061 |
| pcxmi_on_full_context_output | +0.533 | +0.444 |
| pcxmi_on_no_context_output | -0.381 | -0.441 |

Segments: en→zh 34, zh→en 25.

## Free-model replacement for the paid LLM judge (GEMBA / ContextMQM)

`scripts/run_context_llm.py` was run for en↔zh (`--lp zh`, 63 segments of run1's context-COMET MBR outputs) against a
local OpenAI-compatible endpoint (`scripts/serve_openai_compatible.py`) serving `tiny-chatml-lm`:
`gemba_free_endpoint_plumbing_test.csv`. The client, prompt construction with bilingual context, fallback to the fixed
en-de 1-shot example (no en-zh example exists), response parsing and CSV output all worked. The tiny model cannot
produce MQM annotations (all scores 0), so **judge quality was not evaluated**; use an actual open instruct model
(e.g. `vllm serve Qwen/Qwen2.5-72B-Instruct` + `--base_url`).

## Files

`runN/pipeline_report.json` (all scores of the run), `runN/pcxmi_summary.json`, `runN/checks.txt`,
`runN/tower_eval_evaluation.json`, `runN/pipeline_stages.log`, `smoke_lm_*` (smoke-model config and training logs).


## Run 4: the fine-tuned workflow (LoRA fine-tuning, merge, `_empty_sys` pipeline, results table, judge, one-cell translator)

`scripts/run_zh_finetune_smoke_test.sh` (results in `run4_finetuned/`) tests the code behind `kaggle/train_eval_zh_kaggle.ipynb`,
`judge_zh_kaggle.ipynb` and `translate_zh_kaggle.ipynb` with the same tiny models — again **plumbing only, the scores are meaningless**:

```bash
OUT=runs/ft_smoke LM=smoke_models/tiny-chatml-lm COMET_CKPT=smoke_models/tiny-comet/checkpoints/model.ckpt NPROC=2 \
  bash scripts/run_zh_finetune_smoke_test.sh
```

* LoRA fine-tuning with **two processes** (gloo; the multi-GPU path of the Kaggle notebook): loss masking on the answer, time-aware learning-rate schedule,
  synchronised stopping; the merged checkpoint gives the same logits as the adapter loaded on the fly (difference ~1e-5).
* The pipeline with `PROMPT_SUFFIX=_empty_sys` (greedy, candidates, MBR variants, contrastive decoding with one model load for all variants, P-CXMI):
  `run4_finetuned/checks.txt`, **48/48 checks passed**.
* `scripts/consolidate_zh_results.py`: the table with 10 systems and per-message COMET (`results_table.md`).
* The judge against a **fake** API server (`scripts/test_judge_stub.py`): every first request answered with a rate-limit error and retried, answers with markdown
  headers / numbered lists / inline errors parsed, an unparsable answer gives NaN, a second run is answered completely from the cache.
  **No real Gemini request was made.**
* `scripts/test_translate_chat_pipeline.py`: the COMET context strings are identical to the paper code's `add_context_across` (70/70 turn/window/mode combinations),
  concise and verbose output, chrF fallback, typing loop.
