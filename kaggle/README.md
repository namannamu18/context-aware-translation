# Running the en↔zh pipeline on Kaggle (free GPUs)

## Which notebook?

| Notebook | Accelerator | What it does | Time (estimate, not measured on Kaggle yet) |
|---|---|---|---|
| `quick_demo_kaggle.ipynb` | GPU T4 x2 | type messages, get greedy translations with/without context (base model) | 10–15 min |
| `run_zh_pipeline_kaggle.ipynb` | GPU T4 x2 | the paper's pipeline for the **base** model (no fine-tuning) on a small test set | 30–60 min |
| **`train_eval_zh_kaggle.ipynb`** | GPU T4 x2 | **one time:** LoRA fine-tuning on BMELD-train + the paper's pipeline for the fine-tuned model + one results table | about 2.5–3 h with the defaults |
| `judge_zh_kaggle.ipynb` | **None (CPU)** | scores the outputs with Gemini (free API key), adds an MQM column to the results table | 30–40 min (free-tier rate limit) |
| **`translate_zh_kaggle.ipynb`** | GPU T4 x2 | **one cell:** type a conversation, get the fine-tuned system's translations (paper's primary system), optionally graded by the Gemini judge afterwards | a few minutes to load, seconds per message |

## How the files are connected

```
train_eval_zh_kaggle.ipynb  (run once, Save & Run All)
   |  writes zh_lora/        the LoRA adapter + train_info.json (which base model, which prompt format)
   |  writes run_zh/results_zh/...   the results table (test translations of every system, COMET)
   |
   |-- Add Input -> Notebook Output -->  translate_zh_kaggle.ipynb   READS ONLY zh_lora/  (found automatically under /kaggle/input)
   |                                        base model + prompt format come from the adapter folder, so nothing is repeated by hand
   |                                        optional: Gemini judge AFTER each translation (secret GEMINI_API_KEY)
   |
   '-- Add Input -> Notebook Output -->  judge_zh_kaggle.ipynb       READS ONLY run_zh/results_zh/ (found automatically)
                                            adds an MQM score column to the results table; has no effect on translations
```

* Everything runs the same scripts from the repository (each notebook clones `master`): `translate_chat_pipeline.py` (translator) uses `translate_chat.py`,
  `run_context_comet_mbr.py` (the paper's MBR code) and `judge_chat.py` (which uses `run_context_llm.py`, the paper's judge code).
* The translator needs **none** of the evaluation files and does not read the results table. The two notebooks that read training output (`translate` and `judge`) need
  that notebook to have been saved (Save & Run All does it) and added as input.
* `quick_demo_kaggle.ipynb` and `run_zh_pipeline_kaggle.ipynb` are the older base-model notebooks and are not connected to the others.

## The fine-tuned system, step by step

1. Merge the pull request that contains these notebooks (they clone the `master` branch of the repository).
2. Open `train_eval_zh_kaggle.ipynb` → Accelerator **GPU T4 x2**, Internet **On** → **Save Version → Save & Run All (Commit)**. It runs in the background
   (you can close the browser). Settings are in cell 4 (`TRAIN_MINUTES`, `EVAL_DOCS`, `N_CANDIDATES`, `TRAIN_GPUS`). Cell 6 is a **pre-flight** (about 20 minutes): it runs every later step once at a tiny scale with the real models
   (4-step training, the pipeline on 2 conversations, the consistency checks, the results table, the translator), so a problem shows up in the first half hour
   instead of after an hour of training; if it fails, the notebook stops there. After the training, a failing evaluation step is reported at the end but does not stop
   the notebook, so the adapter is never lost. Expect about 3 hours in total. If the 2-GPU launch hangs or fails, set `TRAIN_GPUS = 1`.
   The notebook's **output** keeps the adapter (`zh_lora/`, about 0.2 GB) and the results (`run_zh/`, `results_zh.zip`); the roughly 30 GB of temporary model
   copies disappear when the session ends.
3. *(optional)* `judge_zh_kaggle.ipynb`: *Add Input → Notebook Output →* step 2's notebook; create a free key at <https://aistudio.google.com/apikey> and store it with
   *Add-ons → Secrets* as `GEMINI_API_KEY` (never paste it into a cell or a chat). Run on **no accelerator**: the 30–40 minutes do not use GPU quota.
   Which judge? Gemini's free API tier is the only free option of the two you asked about: a ChatGPT Pro *subscription* does not include OpenAI API access
   (the API is billed separately). Free-tier limits and model names change; the notebook has `JUDGE_MODEL` and `--rpm` settings, and the judge is
   a different model than the paper's GPT-4, so its scores are not comparable to the paper's.
4. `translate_zh_kaggle.ipynb`: *Add Input → Notebook Output →* step 2's notebook → run the two cells; the second one asks for your conversation. Messages start with
   `en:` or `zh:` (several can be pasted on one line); each one is translated with the earlier messages as context. Nothing is trained or evaluated again. **The judge also works here:** with the `GEMINI_API_KEY` secret attached, Gemini grades every translation *after* it was produced (it only grades; the choice among the candidates is made by COMET before). Without the secret the cell runs without the judge; type `judge` to switch it off/on.

Everything here was tested on CPU with tiny stand-in models (`scripts/run_zh_finetune_smoke_test.sh`); the first run with the real models may need small fixes.

---

# The base-model pipeline notebook and the quick demo

> **Just want to see it work?** Import `kaggle/quick_demo_kaggle.ipynb` instead (about 10–15 minutes): it loads
> TowerInstruct-7B once, checks 3 real BMELD conversations in both directions (with/without context), and then lets
> you type chat messages (`en: ...` / `zh: ...`) and get context-aware translations. Same settings: GPU T4 x2, Internet On.
> Import URL: `https://github.com/namannamu18/context-aware-translation/blob/master/kaggle/quick_demo_kaggle.ipynb`

`run_zh_pipeline_kaggle.ipynb` runs the full non-paid pipeline (TowerInstruct-7B-v0.2 + COMET-22) on Kaggle's free
**2× T4** GPUs.

## One-time setup

1. Create a Kaggle account at https://www.kaggle.com and **verify your phone number**
   (Settings → Phone verification). Without it, notebooks cannot use GPUs or the internet.
2. If this GitHub repository is **private**: create a GitHub token with read access to it
   (GitHub → Settings → Developer settings → Fine-grained tokens → *Contents: Read-only*). You'll add it to Kaggle in step 4.

## Open the notebook

1. Kaggle → **Create → New Notebook**.
2. **File → Import Notebook** → *GitHub* tab → paste `https://github.com/namannamu18/context-aware-translation` →
   pick `kaggle/run_zh_pipeline_kaggle.ipynb`. (Or download the `.ipynb` from GitHub and upload it in the *File* tab.)
3. Right-hand panel → **Session options**: *Accelerator* = **GPU T4 x2**, *Internet* = **On**.
4. Private repo only: **Add-ons → Secrets → Add secret**, label `GITHUB_TOKEN`, value = your token, and tick it for this notebook.

## Run

Run the cells in order:

| Cell | What it does |
|---|---|
| 1 | Shows the GPUs (should list two Tesla T4) |
| 2 | Sparse-clones only the code and the en↔zh data |
| 3 | Installs vLLM 0.6.4, COMET 2.2.7, … (if the next cells fail on a torch/CUDA import, **Run → Restart session** and continue from cell 4) |
| 4 | Settings: float16 + tensor parallelism over the two T4s (`VLLM_ENGINE_ARGS`), same for contrastive decoding (`CD_EXTRA_ARGS`) |
| 5 | **Small test run**: 5 BMELD-test conversations, 8 candidates |
| 6 | Consistency checks (should end with "N/N checks passed") |
| 7 | Prints greedy / MBR / contrastive / P-CXMI results |
| 7b–7c | **Type your own messages** and get translations from the full method: greedy with/without context, 16 sampled candidates, best one picked by COMET (also context-aware COMET). Loads the model and COMET once (a few minutes); skipped in *Save & Run All* |
| 8–9 | A bigger run, and a zip of all results for download |

Interactive sessions stop when idle. For long runs use **Save Version → Save & Run All (Commit)**: it runs in the
background for up to 12 h, and everything in `/kaggle/working` is kept as the notebook's output.

## Budget

Kaggle gives about 30 GPU hours per week, with a 12-hour limit per session. The paper setting (100 candidates,
context windows 0/2/6/10/15, full test set of 2,601 segments) does not fit into one session. Split it with
`SKIP_DOCS`/`MAX_DOCS` (e.g. 70 conversations per run) or reduce `N_CANDIDATES`.

## Notes

- T4s have no bfloat16, so the model runs in float16. This can change outputs slightly compared with the paper's bfloat16 runs.
- These settings could not be tested on Kaggle while they were written (the development environment had no GPU and
  no HuggingFace access). If something fails, the error message of the failing step is printed in cell 5's output.
