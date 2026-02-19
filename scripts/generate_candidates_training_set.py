import sys

import pandas as pd
from tower_eval.utils import write_lines
from vllm import LLM, SamplingParams

lp = sys.argv[1]

dataset_path = "/mnt/data/jpombal/axolotl/chat_mt_w_context"

blocks = pd.read_json(dataset_path, lines=True)

blocks["lp"] = blocks["source_language"] + "-" + blocks["target_language"]

model = LLM(
    model="Unbabel/TowerInstruct-WMT24-Chat-7B",
    seed=42,
    gpu_memory_utilization=0.9,
)
tokenizer = model.get_tokenizer()
# epsilon sampling
s = SamplingParams(stop=None, max_tokens=1024, temperature=0.7, min_p=0.02)
N_CANDIDATES = 100


print(f"Generating candidates for {lp}")
instructions = blocks[blocks["lp"] == lp]["instructions"].tolist()
instructions = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": i}], add_generation_prompt=True, tokenize=False
    )
    for i in instructions
]
candidates = []
for l in instructions:
    candidates.extend([l] * N_CANDIDATES)
model_output = model.generate(candidates, s, use_tqdm=True)
generations = [output.outputs[0].text for output in model_output]
write_lines(
    f"/mnt/data/jpombal/wmt24-chat-translation/candidates/training_set/TowerChat/{lp}/{N_CANDIDATES}_epsilon_candidates.txt",
    generations,
    escape_newline=True,
)
