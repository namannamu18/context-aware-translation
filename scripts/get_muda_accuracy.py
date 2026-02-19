import argparse
import json
import os
import sys
from typing import Any, Callable, List

import pandas as pd

sys.path.append(os.environ["MUDA_HOME"])
from muda.langs import create_tagger
from muda.metrics import compute_metrics


def read_file(fname):
    output = []
    with open(fname) as f:
        for line in f:
            output.append(line.strip())
    return output


def recursive_map(func: Callable[[Any], Any], obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: recursive_map(func, v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_map(func, v) for v in obj]
    else:
        return func(obj)


def get_muda_accuracy_score(
    srcs,
    refs,
    docids,
    tgt_lang="de",
    awesome_align_model="bert-base-multilingual-cased",
    awesome_align_cachedir=None,
    load_refs_tags_file=None,
    cohesion_threshold=3,
    dump_hyps_tags_file=None,
    dump_refs_tags_file=None,
    dump_stats_file=None,
    phenomena=["lexical_cohesion", "formality", "verb_form", "pronouns"],
    hyps=None,
) -> None:

    tagger = create_tagger(
        tgt_lang,
        align_model=awesome_align_model,
        align_cachedir=awesome_align_cachedir,
        cohesion_threshold=cohesion_threshold,
    )

    if not load_refs_tags_file:
        preproc = tagger.preprocess(srcs, refs, docids)
        tagged_refs = []
        for doc in zip(*preproc):
            tagged_doc = tagger.tag(*doc, phenomena=phenomena)
            tagged_refs.append(tagged_doc)
    else:
        tagged_refs = json.load(open(load_refs_tags_file))

    preproc = tagger.preprocess(srcs, hyps, docids)
    tagged_hyps = []
    for doc in zip(*preproc):
        tagged_doc = tagger.tag(*doc, phenomena=phenomena)
        tagged_hyps.append(tagged_doc)

    tag_prec, tag_rec, tag_f1 = compute_metrics(tagged_refs, tagged_hyps)
    stat_dicts = []
    for tag in tag_f1:
        print(
            f"{tag} -- Prec: {tag_prec[tag]:.2f} Rec: {tag_rec[tag]:.2f} F1: {tag_f1[tag]:.2f}"
        )
        stat_dicts.append(
            {
                "tag": tag,
                "precision": tag_prec[tag],
                "recall": tag_rec[tag],
                "f1": tag_f1[tag],
            }
        )
    with open(dump_stats_file, "w") as f:
        for d in stat_dicts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    if dump_hyps_tags_file:
        with open(dump_hyps_tags_file, "w", encoding="utf-8") as f:
            json.dump(recursive_map(lambda t: t._asdict(), tagged_hyps), f, indent=2)

    if not load_refs_tags_file and dump_refs_tags_file:
        with open(dump_refs_tags_file, "w", encoding="utf-8") as f:
            json.dump(recursive_map(lambda t: t._asdict(), tagged_refs), f, indent=2)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument(
        "--model",
        help="Model name to get hypothesis in input df.",
        default=None
    )
    parser.add_argument(
        "--hyp_file",
        help="Path to hypothesis file.",
    )

    # comet22 arguments
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--gpus", default=1, type=int)

    # context-comet-qe arguments
    parser.add_argument("--ws", default=2, type=int)

    # MuDA arguments as defined in https://github.com/CoderPat/MuDA/blob/main/muda/main.py
    parser.add_argument("--tgt-lang", type=str, required=True)
    parser.add_argument(
        "--dump_hyps_tags_file",
        type=str,
        default=None,
        help="If set, dumps the hypothesis tags to the specified file.",
    )
    parser.add_argument(
        "--dump_refs_tags_file",
        type=str,
        default=None,
        help="If set, dumps the reference tags to the specified file.",
    )
    parser.add_argument(
        "--dump_stats_file",
        type=str,
        default=None,
        help="If set, dumps stats for each tag to the specified file.",
    )
    parser.add_argument(
        "--load_refs_tags_file",
        type=str,
        default=None,
        help="If set, loads the reference tags from the specified file.",
    )
    parser.add_argument(
        "--cohesion-threshold",
        default=3,
        type=int,
        help="Threshold for number of (previous) occurances to be considered lexical cohesion."
        "Default: 3",
    )
    parser.add_argument(
        "--phenomena",
        nargs="+",
        default=["lexical_cohesion", "formality", "verb_form", "pronouns"],
        help="Phenomena to tag. By default, all phenomena are tagged.",
    )
    parser.add_argument(
        "--awesome-align-model",
        default="bert-base-multilingual-cased",
        help="Awesome-align model to use. Default: bert-base-multilingual-cased",
    )
    parser.add_argument(
        "--awesome-align-cachedir",
        default=None,
        help="Cache directory to save awesome-align models",
    )
    parser.add_argument
    args = parser.parse_args()
    return args


def main(args):
    df = pd.read_csv(args.input_csv)

    if args.model is not None:
        df["hypothesis"] = df[args.model].to_list()
    else:
        df["hypothesis"] = read_file(args.hyp_file)
        
    # MuDA accuracy score
    df = df[df.source_language == "en"]
    get_muda_accuracy_score(
        df["source"].to_list(),
        df["reference"].to_list(),
        df["doc_id"].to_list(),
        hyps=df["hypothesis"].to_list(),
        tgt_lang=args.tgt_lang,
        awesome_align_model=args.awesome_align_model,
        awesome_align_cachedir=args.awesome_align_cachedir,
        dump_hyps_tags_file=args.dump_hyps_tags_file,
        dump_refs_tags_file=args.dump_refs_tags_file,
        dump_stats_file=args.dump_stats_file,
        load_refs_tags_file=args.load_refs_tags_file,
        phenomena=args.phenomena,
        cohesion_threshold=args.cohesion_threshold,
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
