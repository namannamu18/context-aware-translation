# Running the en↔zh pipeline on Kaggle (free GPUs)

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
