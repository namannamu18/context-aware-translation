from tower_eval.utils import read_lines, write_lines
from vllm import LLM, SamplingParams

model = LLM(
    model="Unbabel/TowerInstruct-7B-v0.2",
    seed=42,
    gpu_memory_utilization=0.9,
)
# epsilon sampling
s = SamplingParams(stop=None, max_tokens=1024, temperature=0.7, min_p=0.02)
N_CANDIDATES = 100

prompts = ["no_context", "full_context"]

for dataset in ["bcontrast_test", "wmt24_chat_test"]:
    if dataset == "bcontrast_test":
        lps = [
            "en-de",
            "de-en",
        ]
    else:
        lps = [
            "en-de",
            "de-en",
            "en-fr",
            "fr-en",
            "en-ko",
            "ko-en",
            "en-nl",
            "nl-en",
            "en-pt-br",
            "pt-br-en",
        ]
    for p in prompts:
        for lp in lps:
            print(f"Generating candidates for {dataset} {p} {lp}")
            instructions = read_lines(
                f"/mnt/data/jpombal/wmt24-chat-translation/instructions/{p}/mt/{dataset}.{lp}/instructions.txt",
                unescape_newline=True,
            )
            candidates = []
            for l in instructions:
                candidates.extend([l] * N_CANDIDATES)
            model_output = model.generate(candidates, s, use_tqdm=True)
            generations = [output.outputs[0].text for output in model_output]
            write_lines(
                f"/mnt/data/jpombal/wmt24-chat-translation/candidates/{p}/TowerInstruct-7B-v0.2/{dataset}.{lp}/{N_CANDIDATES}_epsilon_candidates.txt",
                generations,
                escape_newline=True,
            )
