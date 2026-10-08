
import argparse
import pandas as pd
from typing import List, Optional
import torch
from tqdm import tqdm
import torch.nn.functional as F
import sacrebleu
from comet import download_model, load_from_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, LogitsProcessorList, LogitsProcessor

# COMET is only needed with --eval; it is loaded lazily in main() (--comet_model).
comet_metric = None
DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

class LLaMaTranslationModel():
    def __init__(self, model_name_or_path: str, padding: str = "before_system_prompt", torch_dtype: str = "bfloat16", device_map: str = None):
        self.model_name_or_path = model_name_or_path
        # device_map (e.g. "auto") spreads the model over several GPUs, e.g. 2x T4 on Kaggle;
        # torch_dtype float16 for GPUs without bfloat16 support (T4/P100)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                        #   attn_implementation="flash_attention_2",
                                                          torch_dtype=getattr(torch, torch_dtype),
                                                          **({"device_map": device_map} if device_map else {}))
        if not device_map:
            self.model.to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.pipeline = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, return_full_text=False,
                                 **({} if device_map else {"device": DEVICE}))
        assert padding in ["before_system_prompt", "after_system_prompt"]
        self.padding = padding

def load_translation_model(model_name_or_path: str, torch_dtype: str = "bfloat16", device_map: str = None):
    return LLaMaTranslationModel(model_name_or_path=model_name_or_path, torch_dtype=torch_dtype, device_map=device_map)

class EnsembleLogitsProcessor(LogitsProcessor):

    def __init__(self, num_beams: int, 
                 source_weights: List[float] = None, 
                 preserve_bos_token: bool = False,
                 use_entropy_weights: bool = False,
                 use_logits: bool = False,
                 top_k: int = None):
        self.num_beams = num_beams
        self.source_weights = source_weights
        self.preserve_bos_token = preserve_bos_token
        self.use_entropy_weights = use_entropy_weights
        self.use_logits = use_logits
        self.top_k = top_k

    def update_weights_entropy(self, scores):
        # confirm that scores are only for a pair.
        
        # get top_k scores and compute entropy using only those tokens
        if self.top_k:
            scores_topk, _ = torch.topk(scores, self.top_k, dim=-1)
            scores_topk = F.softmax(scores_topk, dim=-1)
        else:
            scores_topk = scores
        entropies = -torch.sum(scores_topk * torch.log(scores_topk + 1e-10), axis=-1) 
        inverse_entropies = 1 / (entropies + 1e-10) 
        total_inverse_entropy = torch.sum(inverse_entropies)

        return inverse_entropies / total_inverse_entropy

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if self.preserve_bos_token and cur_len <= 1:
            return scores

        if not self.use_logits or self.use_entropy_weights:
            scores = F.softmax(scores, dim=-1)

        batch_size = int(input_ids.size(0) / self.num_beams)
        
        if self.source_weights is not None:
            assert len(self.source_weights) == batch_size
            source_weights = torch.Tensor(self.source_weights).to(scores.device)
        else:
            source_weights = 1/(batch_size-1) * torch.ones((batch_size,), device=scores.device)

        for i in range(self.num_beams):
            beam_indices = self.num_beams * torch.arange(batch_size, device=scores.device, dtype=torch.long) + i
            cands = scores[beam_indices]

            if self.use_entropy_weights:
                source_weights = torch.Tensor(self.update_weights_entropy(cands)).to(scores.device)

            if self.use_logits:
                mean_scores = torch.log(F.softmax(source_weights.unsqueeze(-1).expand(-1, scores.size(-1)) * cands, dim=-1).sum(dim=0))
            else:
                mean_scores = torch.log((source_weights.unsqueeze(-1).expand(-1, scores.size(-1)) * cands).sum(dim=0))
            
            for j in beam_indices:
                scores[j] = mean_scores

        if torch.isnan(scores).any():
            scores = torch.nan_to_num(scores, nan=float('-inf'))

        return scores

def translate_multi_source(model, prompts: List[str],
                            src_weights: Optional[List[float]] = None,
                            num_beams: int = 1,
                            max_prompt_length: int = 2048,
                            max_length: int = 4096,
                            use_entropy_weights: bool = False,
                            use_logits: bool = False,
                            sample: bool = False,
                            temperature: float =  1.0,
                            min_p: float = 1.0,
                            entropy_top_k: int = None,
                            **kwargs) -> str:

    inputs = [model.pipeline.preprocess(prompt) for prompt in prompts]
    input_ids = [x['input_ids'][0].tolist() for x in inputs]
    attention_mask = [x['attention_mask'][0].tolist() for x in inputs]

    input_ids = [x[:max_prompt_length] for x in input_ids]
    attention_mask = [x[:max_prompt_length] for x in attention_mask]

    # Llama/Tower sentencepiece tokenizers: pad with "▁" (original behaviour); other tokenizers: pad token
    pad_token_id = model.tokenizer.get_vocab().get("▁", model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id)
    max_len = max(len(x) for x in input_ids)
    if model.padding == "before_system_prompt":
        input_ids = [[pad_token_id] * (max_len - len(x)) + x for x in input_ids]
        attention_mask = [[0] * (max_len - len(x)) + x for x in attention_mask]
    elif model.padding == "after_system_prompt":
        sys_end_id = model.tokenizer.get_vocab()[">>"]
        for i in range(len(input_ids)):
            second_inst_idx = input_ids[i].index(sys_end_id, 1)
            input_ids[i] = (input_ids[i][:second_inst_idx + 1] +
                            [pad_token_id] * (max_len - len(input_ids[i])) +
                            input_ids[i][second_inst_idx + 1:])
            attention_mask[i] = (attention_mask[i][:second_inst_idx + 1] +
                                    [0] * (max_len - len(attention_mask[i])) +
                                    attention_mask[i][second_inst_idx + 1:])

    input_ids = torch.tensor(input_ids).to(model.model.device)
    attention_mask = torch.tensor(attention_mask).to(model.model.device)
    logits_processor = LogitsProcessorList([
        EnsembleLogitsProcessor(num_beams=num_beams, source_weights=src_weights, use_entropy_weights=use_entropy_weights, use_logits=use_logits, top_k=entropy_top_k),
    ])
    output = model.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=num_beams,
        eos_token_id=model.tokenizer.eos_token_id,
        max_length=max_length, 
        logits_processor=logits_processor,
        remove_invalid_values=True,
        # Disable sampling
        do_sample=sample,
        temperature=temperature,
        min_p=min_p,
        **kwargs,
    )
    
    output = output.reshape(1, output.shape[0], *output.shape[1:])
    output = {
        "generated_sequence": output,
        "input_ids": input_ids[0],
        "prompt_text": prompts[0],
    }
    output = model.pipeline._ensure_tensor_on_device(output, device=torch.device("cpu"))
    output = model.pipeline.postprocess(output)
    output = output[0]['generated_text']
    _, output = output.rsplit("\n", maxsplit=1)
    return output

def read_file(fname, unescape_newline=True):
    output = []
    with open(fname) as f:
        for line in f:
            output.append(line.strip())
    if unescape_newline:
      output = [l.replace("\\n", "\n") for l in output]
    return output

def get_instructions(inst_dir, df, lp, context_type, split="dev", data_name="wmt24_chat"):
  source_lang = lp.split("-")[0]
  target_lang = lp.split("-")[1].replace('pt','pt-br')

  xx_yy_insts = read_file(f"{inst_dir}/{context_type}/mt/{data_name}_{split}.{source_lang}-{target_lang}/instructions.txt")
  yy_xx_insts = read_file(f"{inst_dir}/{context_type}/mt/{data_name}_{split}.{target_lang}-{source_lang}/instructions.txt")

  assert len(xx_yy_insts) + len(yy_xx_insts) == len(df)
  instructions = []
  j,k=0,0

  for i, row in df.iterrows():
    if row["lp"] == f"{source_lang}-{target_lang}":
      instructions.append(xx_yy_insts[j])
      j+=1
    else:
      # nl-en and it-en are considered here
      instructions.append(yy_xx_insts[k])
      k+=1
  return instructions

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--lang_pair", type=str, default="en-de")
    parser.add_argument("--context_weight", type=float, default=1.)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--non_context_weight", type=float, default=1.)
    parser.add_argument("--model_name_or_path", type=str, default="TowerInstruct-v0.2-w-chat-mt-data")
    parser.add_argument("--instructions_dir", type=str, default="instructions")
    parser.add_argument("--data_dir", type=str, default="paper_results")
    parser.add_argument("--save_output", type=str, default=None)
    parser.add_argument("--use_entropy_weights", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--sample", action='store_true')
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--use_logits", action='store_true')
    parser.add_argument("--entropy_top_k", type=int, default=None)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--data_name", type=str, default="wmt24_chat", help="Instructions are read from <instructions_dir>/<prompt>/mt/<data_name>_<split>.<lp>")
    parser.add_argument("--context_prompt", type=str, default="full_context_empty_sys")
    parser.add_argument("--no_context_prompt", type=str, default="no_context_empty_sys")
    parser.add_argument("--comet_model", type=str, default="Unbabel/wmt22-comet-da", help="HF id or local .ckpt (only used with --eval)")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device_map", type=str, default=None, help='e.g. "auto" to split the model over all visible GPUs')
    parser.add_argument("--max_new_tokens", type=int, default=None, help="If set, caps generated tokens (takes precedence over --max_length)")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Run several settings with ONE model load, named like the files of the paper: c<context_weight>_nc<non_context_weight>"
                             "[_b<beams>][_t<temp>_minp<min_p>_n<n_samples>], e.g. c1_nc1 c5_nc1 c1_nc1_t0.7_minp0.02_n4. Outputs: <save_dir>/<variant>.out.txt")
    parser.add_argument("--save_dir", type=str, default=None, help="Output folder for --variants")
    args = parser.parse_args()
    return args

def parse_variant(name):
    """c<cw>_nc<ncw>[_b<beams>][_t<temp>_minp<min_p>_n<n>] -> decoding settings (the names of the files in contrast_decode/)."""
    import re
    m = re.fullmatch(r"c([\d.]+)_nc([\d.]+)(?:_b(\d+))?(?:_t([\d.]+)_minp([\d.]+)_n(\d+))?", name)
    if not m:
        raise ValueError(f"cannot parse contrastive decoding variant '{name}'")
    cw, ncw, beams, temp, min_p, n = m.groups()
    cfg = {"context_weight": float(cw), "non_context_weight": float(ncw), "num_beams": int(beams) if beams else 1,
           "sample": temp is not None, "temperature": float(temp) if temp else 1.0, "min_p": float(min_p) if min_p else 0.0,
           "num_return_sequences": int(n) if n else 1}
    return cfg


def decode(model, context_ints, non_context_ints, cfg, args):
    all_pairs = []
    for (x, y) in list(zip(context_ints, non_context_ints)):
        all_pairs.extend([(x, y)] * cfg["num_return_sequences"])

    translations = []
    for pair in tqdm(all_pairs):
        translation = translate_multi_source(
            model,
            src_weights=[cfg["context_weight"], cfg["non_context_weight"]],
            max_length=args.max_length,
            prompts=pair,
            max_prompt_length=args.max_prompt_length,
            use_entropy_weights=args.use_entropy_weights,
            use_logits=args.use_logits,
            num_beams=cfg["num_beams"],
            sample=cfg["sample"],
            temperature=cfg["temperature"],
            min_p=cfg["min_p"],
            entropy_top_k=args.entropy_top_k,
            **({"max_new_tokens": args.max_new_tokens} if args.max_new_tokens else {}),
            )
        translations.append(translation)
    return translations


def evaluate_and_save(df, translations, save_output, args):
    """Write <save_output>.out.txt and, with --eval, the COMET / chrF scores (<save_output>.score.txt)."""
    global comet_metric
    if save_output is not None:
        with open(save_output + '.out.txt', 'w') as f:
            for line in translations:
                f.write(f"{line}\n")

    if args.eval and len(translations) == len(df):
        if comet_metric is None:
            comet_metric = load_from_checkpoint(args.comet_model if args.comet_model.endswith(".ckpt") else download_model(args.comet_model))
        df = df.copy()
        df["output-select"] = translations
        df[f"output-comet"]  = comet_metric.predict([{"mt": y, "ref":z, "src": x} for x, y, z in zip(df["source"], df["output-select"], df["reference"])],
                batch_size=64, gpus=1 if torch.cuda.is_available() else 0, progress_bar=True, **({"devices": [0]} if torch.cuda.is_available() else {}))['scores']
        df[f"output-chrf"] = [sacrebleu.sentence_chrf(x, [y]).score for (x, y) in zip(df["output-select"].to_list(), df["reference"].to_list())]

        if save_output is not None:
            with open(save_output + '.score.txt', 'w') as f:
                for lp, lp_df in df.groupby("lp"):
                    print(lp, lp_df[f"output-comet"].mean(), lp_df[f"output-chrf"].mean())
                    f.write(f'Comet: {lp_df["output-comet"].mean()}\n')
                    f.write(f'Chrf: {lp_df["output-chrf"].mean()}\n')


def main(args):
    lang_pair=args.lang_pair

    model = load_translation_model(args.model_name_or_path, args.torch_dtype, args.device_map)

    df = pd.read_csv(f"{args.data_dir}/{args.split}.{lang_pair}.csv")

    context_ints = get_instructions(args.instructions_dir, df, lang_pair, args.context_prompt, args.split, args.data_name)
    non_context_ints = get_instructions(args.instructions_dir, df, lang_pair, args.no_context_prompt, args.split, args.data_name)

    if args.variants:   # several settings, one model load (same names as the files of the paper)
        assert args.save_dir, "--variants needs --save_dir"
        import os
        os.makedirs(args.save_dir, exist_ok=True)
        for name in args.variants:
            print(f"### contrastive decoding variant {name}")
            cfg = parse_variant(name)
            translations = decode(model, context_ints, non_context_ints, cfg, args)
            evaluate_and_save(df, translations, f"{args.save_dir}/{name}", args)
        return

    cfg = {"context_weight": args.context_weight, "non_context_weight": args.non_context_weight, "num_beams": args.num_beams,
           "sample": args.sample, "temperature": args.temperature, "min_p": args.min_p, "num_return_sequences": args.num_return_sequences}
    translations = decode(model, context_ints, non_context_ints, cfg, args)
    evaluate_and_save(df, translations, args.save_output, args)


if __name__ == "__main__":
    args = get_args()
    main(args)