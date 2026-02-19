import json
import os
import torch
import argparse
import logging
import pickle

from alti_analysis.utils import (load_model_tokenizer,load_contributions,load_outputs,load_prep_data,
                                 aggregate_contributions,get_prompt_segments,get_target_prefix_segments,get_gen_seq_segments,
                                 create_directory,aggregate_contributions_normalized)

from alti_analysis.utils import SegmentationClass,ContributionsAggregatorClass

def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    general_group = parser.add_argument_group('General options')
    data_group = parser.add_argument_group('Data options')
    data_group.add_argument('--model_name', choices=['TowerInstruct-7B-v0.2','TowerInstruct-7B-w-chat-empty-sys'],required=True)
    data_group.add_argument('--model_prec',choices=["full","bf16","fp16"], help='Whether to use full precision,bf16 or f16')
    # data_group.add_argument('--model_version', choices=['greedy-7b-w-context', 'greedy-7b-wo-context','greedy-7b-ft-w-context','greedy-7b-ft-wo-context'], required=True)
    data_group.add_argument('--language',type=str,required=True, help='Language direction to run aggregation on')
    data_group.add_argument('--contributions_folder',type=str,required=True, help='Folder to read outputs/contributions.')
    # data_group.add_argument('--data_path',type=str,required=True, help='data path for alti prep data')
    data_group.add_argument('--save_folder',type=str,required=True, help='Folder to save aggregated contributions.')
    data_group.add_argument('--normalized',action='store_true',help='Whether to normalize the contributions (token-length-normalization)')
    data_group.add_argument('--context_aggregation_type',choices=['context-level', 'turn-level','no_context'])
    data_group.add_argument('--num_turns',type=int, default=None, help='Number of turns in case of context_type==fixed_context_window and aggregation_type==turn-level')

    return parser.parse_args()


def pathmaking_contrs(contributions_folder,language):
    """
    This functions returns the paths for writing outputs and computed contributions.
    """
    contributions_path = os.path.join(contributions_folder,language,"contr_alti/contrs.pickle")
    # outs_path = os.path.join(contributions_folder,"outputs/all_outputs.pickle")
    return contributions_path




def main(args):
    logging.info("ALTI CONTRIBUTIONS AGGREGATION SCRIPT")
    model_name = args.model_name
    model_prec = args.model_prec
    num_turns = args.num_turns
    context_aggregation_type = args.context_aggregation_type
    contributions_out_folder = args.contributions_folder
    # data_path = args.data_path
    language = args.language
    source_lang,target_lang = language.split('-')
    save_folder = args.save_folder


    logging.info("Contr path: ",contributions_out_folder)

    #load tokenizer to be used
    if model_name=="TowerInstruct-7B-v0.2":
        logging.info("Loading tower-instruct tokenizer...")
        name_path = "Unbabel/TowerInstruct-7B-v0.2"
    elif model_name == "TowerInstruct-7B-w-chat-empty-sys":
        logging.info("Loading tower-chat tokenizer...")
        # assert False, "TowerChat not implemented yet"
        name_path = "/mnt/data-poseidon/sweta/chat-translation-generation/TowerInstruct-v0.2-w-chat-mt-data"
    else:
        raise ValueError("Unknown model")
    if model_prec == "bf16":
        _, tokenizer = load_model_tokenizer(name_path, torch_dtype=torch.bfloat16,only_tokenizer=True)
    elif model_prec == "fp16":
        _, tokenizer = load_model_tokenizer(name_path, torch_dtype=torch.float16,only_tokenizer=True)
    elif model_prec == "full":
        _, tokenizer = load_model_tokenizer(name_path, torch_dtype=None,only_tokenizer=True)
    else:
        raise IOError("Specify correctly model precision (full or bf16 or f16)")


    contr_path = pathmaking_contrs(contributions_out_folder,language)
    contrs_list,ids_list,input_tokens_list,gen_tokens_list = load_contributions(contr_path)

    input_tokens_list = [elem.squeeze(0) for elem in input_tokens_list]
    gen_tokens_list = [elem.squeeze(0) for elem in gen_tokens_list]

    #FILTERING ACCORDING TO CONTEXT HAS ALREADY BE DONE IN ALTI PREP DATA -- NO NEED TO CHECK THE FOLLOWING:

    # # we load the corresponding prompts and generated outputs (as strings) -- need only for locating samples without context
    # prompts_list,gens_outs_list = load_prep_data(data_path,model_version,keep_ids=ids_list)
    # import ipdb; ipdb.set_trace()
    # # remove any instances that dont have context (usually these are the first turn of each conversation)
    # if "w-context" in model_version:
    #     no_context_indices = [index for index,prompt in enumerate(prompts_list) if "\nContext:" not in prompt]
    #
    #     contrs_list = [contr for index,contr in enumerate(contrs_list) if not index in no_context_indices]
    #     ids_list = [id for index,id in enumerate(ids_list) if not index in no_context_indices]
    #     input_tokens_list = [input_tokens for index,input_tokens in enumerate(input_tokens_list) if  index  not in no_context_indices]
    #     gen_tokens_list = [gen_tokens for index,gen_tokens in enumerate(gen_tokens_list) if  index not in no_context_indices]


    segmentator = SegmentationClass(context_type_segmentation=context_aggregation_type,model_version=model_name,model_tokenizer=tokenizer,num_turns=num_turns)

    aggregator = ContributionsAggregatorClass(normalized_contributions=args.normalized,segmentator=segmentator)
    outputs = aggregator.get_aggregated_contributions(ids_list,contrs_list,input_tokens_list,gen_tokens_list)

    # segmented_prompts = segmentator.get_prompt_segments(input_tokens_list)
    # segmented_tps = segmentator.get_target_prefix_segments([elem[:-1] for elem in gen_tokens_list])
    # segmented_gens = segmentator.get_gen_seq_segments(gen_tokens_list)
    # import ipdb; ipdb.set_trace()

    # prompt_seg_dict_all_samples = get_prompt_segments(model_name, tokenizer, input_tokens_list)
    # tp_seg_dict_all_samples = get_target_prefix_segments([elem[:-1] for elem in gen_tokens_list]) #for the target prefix we exclude the last token!
    # gen_seg_dict_all_samples = get_gen_seq_segments(gen_tokens_list)
    #
    # prompt_segments_all = [sample['segments'] for sample in prompt_seg_dict_all_samples]
    # prompt_segment_lengths_all = [sample['lengths'] for sample in prompt_seg_dict_all_samples]
    # tp_segments_all = [sample['segments'] for sample in tp_seg_dict_all_samples]
    # tp_segment_lengths_all = [sample['lengths'] for sample in tp_seg_dict_all_samples]
    # gen_segments_all = [sample['segments'] for sample in gen_seg_dict_all_samples]
    # gen_segment_lengths_all = [sample['lengths'] for sample in gen_seg_dict_all_samples]

    # if args.normalized:
    #     aggregated_contr_dict = aggregate_contributions_normalized(ids_list,contrs_list,input_tokens_list,gen_tokens_list,prompt_segment_lengths_all,tp_segment_lengths_all,gen_segment_lengths_all)
    #
    # else:
    #     aggregated_contr_dict = aggregate_contributions(ids_list,contrs_list,input_tokens_list,gen_tokens_list,prompt_segment_lengths_all,tp_segment_lengths_all,gen_segment_lengths_all)

    # #store aggregated contributions
    save_folder = os.path.join(save_folder,language,context_aggregation_type)
    create_directory(save_folder)
    with open(os.path.join(save_folder,'aggregated_contributions.pkl'),"wb") as file:
        pickle.dump(outputs["aggregated_contrs"],file,protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save_folder, 'segments.json'), "w") as json_file:
        json.dump(outputs["segments"], json_file, indent=4)
    with open(os.path.join(save_folder, 'ids.json'), "w") as json_file:
        json.dump(outputs["ids"], json_file, indent=4)
    logging.info("finished successfully!")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)