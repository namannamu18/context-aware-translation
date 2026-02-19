# A Context-aware Framework for Translation-mediated Conversations

[Paper (arXiv)](https://arxiv.org/abs/2412.04205) | [WMT 2024 Chat Shared Task Submission](https://aclanthology.org/2024.wmt-1.100.pdf)

## Installation

Install [`tower-eval`](https://github.com/deep-spin/tower-eval) to run generation and evaluation configs.

## Pipeline

Data prep → Candidate generation (epsilon sampling) → Decoding (greedy / MBR / contrastive) → PCXMI analysis → Evaluation

## Languages & Models

**Language pairs** — WMT24 Chat: en↔{de, fr, pt-br, nl, ko}; BConTrasT: en↔de

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
