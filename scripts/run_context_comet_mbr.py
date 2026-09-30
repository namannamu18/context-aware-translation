from comet import download_model, load_from_checkpoint
import pandas as pd
import argparse
import torch
from tqdm import tqdm
import numpy as np
import sacrebleu
from typing import List
from itertools import chain

# The COMET model is loaded in main() (--comet_model, default Unbabel/wmt22-comet-da)
# so that the script can also be used with a local checkpoint.
comet_metric = None
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def load_comet(name):
    checkpoint_path = name if name.endswith(".ckpt") else download_model(name)
    return load_from_checkpoint(checkpoint_path)

def read_file(fname, unescape_newline=True):
    output = []
    with open(fname) as f:
        for line in f:
            output.append(line.strip())
    if unescape_newline:
      output = [l.replace("\\n", "\n") for l in output]
    return output


# Referred from https://github.com/amazon-science/doc-mt-metrics/blob/main/Prism/add_context.py
def add_context_across(orig_txt: List[str], context_same: List[str], context_other: List[str], 
                sender_ids: List[str], sep_token: str = "</s>", ws: int = 2) -> List[str]:
    if not (len(orig_txt) == len(context_same)== len(context_other)):
        raise Exception(f'Lengths should match: len(orig_txt)={len(orig_txt)}, len(context_same)={len(context_same)}, len(context_other)={len(context_other)}')
    context_all = []
    for i in range(len(orig_txt)):
      context_window = []
      for j in range(max(0, i - ws), i):
        if sender_ids[j] == sender_ids[i]:
            context_window.append(context_same[j])
        else:
            context_window.append(context_other[j])
      context_all.append(context_window)

    augm_txt = []
    for i in range(len(orig_txt)):
      context = context_all[i]
      augm_txt.append(" {} ".format(sep_token).join(context + [orig_txt[i]]))

    return augm_txt
   

def build_embeddings(sources, translations, comet_model, batch_size):

    src_batches = [
        sources[i : i + batch_size] for i in range(0, len(sources), batch_size)
    ]
    src_inputs = [comet_model.encoder.prepare_sample(batch) for batch in src_batches]
    mt_batches = [
        translations[i : i + batch_size]
        for i in range(0, len(translations), batch_size)
    ]
    mt_inputs = [comet_model.encoder.prepare_sample(batch) for batch in mt_batches]

    src_embeddings = []
    with torch.no_grad():
        for batch in src_inputs:
            input_ids = batch["input_ids"].to(comet_model.device)
            attention_mask = batch["attention_mask"].to(comet_model.device)
            src_embeddings.append(
                comet_model.get_sentence_embedding(input_ids, attention_mask)
            )
    src_embeddings = torch.vstack(src_embeddings)

    mt_embeddings = []
    with torch.no_grad():
        for batch in tqdm(mt_inputs, desc="Encoding sentences...", dynamic_ncols=True):
            input_ids = batch["input_ids"].to(comet_model.device)
            attention_mask = batch["attention_mask"].to(comet_model.device)
            mt_embeddings.append(
                comet_model.get_sentence_embedding(input_ids, attention_mask)
            )
    mt_embeddings = torch.vstack(mt_embeddings)

    return src_embeddings, mt_embeddings

def get_candidates(df, lp, split="dev", n=100, model_name="TowerInstruct-7B-w-chat-empty-sys", context_type="full_context_empty_sys", data_name="wmt24_chat"):
  # n = number of epsilon-sampling candidates per segment ({n}_epsilon_candidates.txt)
  source_lang = lp.split("-")[0]
  target_lang = lp.split("-")[1].replace('pt','pt-br')
  if split=="train":
     xx_yy_generations = read_file(f"training_set/TowerChat/{source_lang}-{target_lang}/100_epsilon_candidates.txt")
     yy_xx_generations = read_file(f"training_set/TowerChat/{target_lang}-{source_lang}/100_epsilon_candidates.txt")
  else:
    try:
       xx_yy_generations = read_file(f"candidates/{context_type}/{model_name}/{data_name}_{split.replace('test', 'test_blind')}.{source_lang}-{target_lang}/{n}_epsilon_candidates.txt")
       yy_xx_generations = read_file(f"candidates/{context_type}/{model_name}/{data_name}_{split.replace('test', 'test_blind')}.{target_lang}-{source_lang}/{n}_epsilon_candidates.txt")
    except:
       xx_yy_generations = read_file(f"candidates/{context_type}/{model_name}/{data_name}_{split}.{source_lang}-{target_lang}/{n}_epsilon_candidates.txt")
       yy_xx_generations = read_file(f"candidates/{context_type}/{model_name}/{data_name}_{split}.{target_lang}-{source_lang}/{n}_epsilon_candidates.txt")


  xx_yy_generations = np.array(xx_yy_generations).reshape((len(xx_yy_generations)//n, n))
  yy_xx_generations = np.array(yy_xx_generations).reshape((len(yy_xx_generations)//n, n))

  assert len(xx_yy_generations) + len(yy_xx_generations) == len(df)
  translations = []
  j,k=0,0

  for i, row in tqdm(df.iterrows()):
    if row["lp"] == f"{source_lang}-{target_lang}":
      translations.append(xx_yy_generations[j])
      j+=1
    else:
      # nl-en and it-en are considered here
      translations.append(yy_xx_generations[k])
      k+=1
  return np.array(translations)

def get_translations(df, lp, context_type, model_name, split="dev", data_name="wmt24_chat", gen_backend="vllm"):
  source_lang = lp.split("-")[0]
  target_lang = lp.split("-")[1].replace('pt','pt-br')

  try:
    xx_yy_generations = read_file(f"generations/{context_type}/mt/{data_name}_{split.replace('test', 'test_blind')}.{source_lang}-{target_lang}/{gen_backend}/{model_name}/generation.txt")
    yy_xx_generations = read_file(f"generations/{context_type}/mt/{data_name}_{split.replace('test', 'test_blind')}.{target_lang}-{source_lang}/{gen_backend}/{model_name}/generation.txt")
  except:
    xx_yy_generations = read_file(f"generations/{context_type}/mt/{data_name}_{split}.{source_lang}-{target_lang}/{gen_backend}/{model_name}/generation.txt")
    yy_xx_generations = read_file(f"generations/{context_type}/mt/{data_name}_{split}.{target_lang}-{source_lang}/{gen_backend}/{model_name}/generation.txt")
   
  assert len(xx_yy_generations) + len(yy_xx_generations) == len(df)
  translations = []
  j,k=0,0

  for i, row in df.iterrows():
    if row["lp"] == f"{source_lang}-{target_lang}":
      translations.append(xx_yy_generations[j])
      j+=1
    else:
      # nl-en and it-en are considered here
      translations.append(yy_xx_generations[k])
      k+=1
  return translations


def run_mbr_chrf(sources, translations, n_sent, num_samples):
    """MBR with chrF as the utility function. Only meant for environments where
    no COMET checkpoint is available (the method uses COMET, see run_mbr)."""
    translations = np.array(translations, dtype=object).reshape(n_sent, num_samples)
    mbr_matrix = np.zeros((n_sent, num_samples))
    for i in tqdm(range(n_sent), desc="MBR Scores (chrF)...", dynamic_ncols=True):
        for j in range(num_samples):
            scores = [sacrebleu.sentence_chrf(translations[i, j], [translations[i, k]]).score for k in range(num_samples) if k != j]
            mbr_matrix[i, j] = np.mean(scores)
    return mbr_matrix


def run_mbr(comet_metric, sources, translations, n_sent, num_samples, batch_size=64, use_context=False, device_id=DEVICE):
    
    if use_context:
        comet_metric.enable_context()
    else:
        comet_metric.use_context = False

    comet_metric.eval()
    comet_metric.to(device_id)

    src_embeddings, mt_embeddings = build_embeddings(
        sources, translations, comet_metric, batch_size
    )

    src_embeddings = src_embeddings.reshape(n_sent, num_samples, -1)

    mt_embeddings = mt_embeddings.reshape(n_sent, num_samples, -1)
    mbr_matrix = torch.zeros(n_sent, num_samples)

    with torch.no_grad():
        # Loop over all source sentences
        for i in tqdm(
            range(mbr_matrix.shape[0]), desc="MBR Scores...", dynamic_ncols=True
        ):
            source = src_embeddings[i, :]
            # Loop over all hypothesis
            for j in range(mbr_matrix.shape[1]):
                translation = mt_embeddings[i, j, :].repeat(num_samples, 1)
                # Score current hypothesis against all others
                pseudo_refs = mt_embeddings[i, :]
                scores = comet_metric.estimate(source, translation, pseudo_refs)["score"]
                scores = torch.cat([scores[0:j], scores[j + 1 :]])
                mbr_matrix[i, j] = scores.mean()
    return mbr_matrix.cpu().detach().numpy()

def get_source_and_translations(df, output_reshaped, use_context, context_size=2, context_source="source", context_mt="source"):
    if use_context:
        outputs_with_context = []
        sources_with_context = []
        for i in range(len(output_reshaped.T)):
            df["out"] = output_reshaped.T[i]
            outputs_curr = []
            src_curr = []
            for _, df_group in df.groupby(["doc_id"], sort=False):
                src_with_context = add_context_across(orig_txt=df_group["source"].to_list(),
                                                context_same=df_group[context_source].to_list(),
                                                context_other=df_group[context_mt].to_list(),
                                                sender_ids=df_group["sender"].to_list(),
                                                sep_token=comet_metric.encoder.tokenizer.sep_token if comet_metric is not None else "</s>",
                                                ws=context_size,)
                out_with_context = add_context_across(orig_txt=df_group["out"].to_list(),
                                                context_same=df_group[context_mt].to_list(),
                                                context_other=df_group[context_source].to_list(),
                                                sender_ids=df_group["sender"].to_list(),
                                                sep_token=comet_metric.encoder.tokenizer.sep_token if comet_metric is not None else "</s>",
                                                ws=context_size,)
                outputs_curr.extend(out_with_context)
                src_curr.extend(src_with_context)
            outputs_with_context.append(outputs_curr)
            sources_with_context.append(src_curr)
                
        sources = list(chain.from_iterable(np.array(sources_with_context).T))
        outputs = list(chain.from_iterable(np.array(outputs_with_context).T))
        
    else:
        outputs = list(chain.from_iterable(output_reshaped))
        sources = list(np.repeat(df["source"].to_list(), output_reshaped.shape[1]))
    
    return sources, outputs

ORIGINAL_DATA_DIR = "/mnt/data-poseidon/sweta/context-aware-mt/chat-translation-generation/wmt24-chat-translation/tacl_results"


def get_dataframe(split, lang_pair, candidate_file, data_name='wmt24_chat', model_name='TowerInstruct-7B-w-chat-empty-sys', context_type="full_context_empty_sys", data_dir=ORIGINAL_DATA_DIR, num_candidates=100, gen_backend="vllm"):
    df = pd.read_csv(f"{data_dir}/{data_name}/{split}.{lang_pair}.csv",
                    keep_default_na=False,
        na_values=["NaN"],)
    df["lp"] = df["source_language"] + "-" + df["target_language"]

    if candidate_file:
        outputs = read_file(candidate_file)
        output_reshaped = np.array(outputs).reshape((len(outputs)//len(df),  len(df))).T
    else:
        output_reshaped = get_candidates(df, lang_pair, split, n=num_candidates, data_name=data_name, model_name=model_name, context_type=context_type)

    if split=="dev" or split=="test":
        df["greedy"] = get_translations(df, lang_pair, 
                                    context_type=context_type, 
                                    model_name=model_name, 
                                    data_name=data_name,
                                    split=split,
                                    gen_backend=gen_backend)
    return df, output_reshaped


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_pair", type=str, default="en-de")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--candidate-file", type=str, default=None)
    parser.add_argument("--use_candidates", action='store_true')
    parser.add_argument("--use_context", action='store_true')
    parser.add_argument("--context_mt", type=str, default="source")
    parser.add_argument("--context_source", type=str, default="source")
    parser.add_argument("--context_size", type=int, default=2)
    parser.add_argument("--model_name", type=str, default='TowerInstruct-7B-w-chat-empty-sys')
    parser.add_argument("--data_name", type=str, default='wmt24_chat')
    parser.add_argument("--inf_context", type=str, default='full_context_empty_sys')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_output", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=ORIGINAL_DATA_DIR, help="Folder with <data_name>/<split>.<lang_pair>.csv (e.g. paper_results_zh for en-zh)")
    parser.add_argument("--num_candidates", type=int, default=100)
    parser.add_argument("--gen_backend", type=str, default="vllm", help="generations/<...>/<gen_backend>/<model_name>/generation.txt")
    parser.add_argument("--comet_model", type=str, default="Unbabel/wmt22-comet-da", help="HF id or local .ckpt")
    parser.add_argument("--utility", type=str, default="comet", choices=["comet", "chrf"],
                        help="MBR utility. comet = the method in the paper; chrf = fallback when no COMET checkpoint is available")
    args = parser.parse_args()
    return args

def main(args):
    global comet_metric
    try:
        comet_metric = load_comet(args.comet_model)
    except Exception as e:
        if args.utility == "comet":
            raise
        print(f"COMET not available ({type(e).__name__}); running chrF-MBR and skipping COMET scores.")
        comet_metric = None

    def mbr(sources, outputs, n_sent, num_samples, use_context):
        if args.utility == "chrf":
            return run_mbr_chrf(sources, outputs, n_sent, num_samples)
        return run_mbr(comet_metric, sources, outputs, n_sent, num_samples, batch_size=args.batch_size, use_context=use_context)

    df, output_reshaped = get_dataframe(args.split, args.lang_pair, args.candidate_file, args.data_name, args.model_name, args.inf_context,
                                        data_dir=args.data_dir, num_candidates=args.num_candidates, gen_backend=args.gen_backend)
    n_sent, num_samples = output_reshaped.shape


    if args.context_mt == "comet-best":
        sources, outputs = get_source_and_translations(df, 
                                                   output_reshaped, 
                                                   False)

        mbr_matrix = mbr(sources, outputs, n_sent, num_samples, use_context=False)
        if comet_metric is not None:
            comet_metric.use_context = False
        col_argmax = np.argmax(mbr_matrix, axis=1)
        df["comet-best"]  = output_reshaped[np.arange(len(col_argmax)), col_argmax]
    
    sources, outputs = get_source_and_translations(df, 
                                                   output_reshaped, 
                                                   args.use_context, 
                                                   args.context_size,
                                                   args.context_source,
                                                   args.context_mt)
    
    mbr_matrix = mbr(sources, outputs, n_sent, num_samples, use_context=args.use_context)

    if comet_metric is not None:
        comet_metric.use_context = False
    col_argmax = np.argmax(mbr_matrix, axis=1)
    df["output-select"]  = output_reshaped[np.arange(len(col_argmax)), col_argmax]

    if args.split == "dev":
        for col in ["output-select", "greedy"]:
            if comet_metric is not None:
                df[f"{col}-comet"]  = comet_metric.predict([{"mt": y, "ref":z, "src": x} for x, y, z in zip(df["source"], df[col], df["reference"])],
                    batch_size=64, gpus=1 if torch.cuda.is_available() else 0, progress_bar=True, **({"devices": [0]} if torch.cuda.is_available() else {}))['scores']
            else:
                df[f"{col}-comet"] = np.nan
            df[f"{col}-chrf"] = [sacrebleu.sentence_chrf(x, [y]).score for (x, y) in zip(df[col].to_list(), df["reference"].to_list())]

            for lp, lp_df in df.groupby("lp"):
                print(col, lp, lp_df[f"{col}-comet"].mean(), lp_df[f"{col}-chrf"].mean())

    if args.save_output is not None:
        df.to_csv(f"{args.save_output}.csv")
        # with open(args.save_output + '.out.txt', 'w') as f:
        #     for line in df["output-select"].to_list():
        #         f.write(f"{line}\n") 
        
        if args.split == "dev":
            with open(args.save_output + '.score.txt', 'w') as f:
                for lp, lp_df in df.groupby("lp"):
                    f.write(f'Comet: {lp_df["output-select-comet"].mean()}\n')
                    f.write(f'Chrf: {lp_df["output-select-chrf"].mean()}\n')

if __name__ == "__main__":
    args = get_args()
    main(args)