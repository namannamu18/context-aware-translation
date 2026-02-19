import torch
import pickle
import json
import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re


def create_directory_structure(fullpath):
    """
    Given a file path, create the directory structure
    """
    directory_name = os.path.dirname(fullpath)
    create_directory(directory_name)

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        pass
        # print(f"Directory '{directory_path}' already exists.")

def load_model_tokenizer(name_path, only_tokenizer=False,torch_dtype=None):
    """Load model and tokenizer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if name_path == 'facebook/opt-125m':
        # use_fast = False as indicated in https://huggingface.co/docs/transformers/model_doc/opt
        tokenizer = AutoTokenizer.from_pretrained(name_path, use_fast=False)
        if only_tokenizer == False:
            model = AutoModelForCausalLM.from_pretrained(name_path)
    elif name_path == 'gpt2' or name_path == 'gpt2-large' or name_path == 'gpt2-xl':
        tokenizer = GPT2Tokenizer.from_pretrained(name_path)
        if only_tokenizer == False:
            model = GPT2LMHeadModel.from_pretrained(name_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(name_path)
        if only_tokenizer == False:
            if name_path=="meta-llama/Llama-2-7b-hf":
                model = AutoModelForCausalLM.from_pretrained(name_path,attn_implementation="eager", token=True,torch_dtype=torch_dtype) #attn_implementation="eager"
            else:
                model = AutoModelForCausalLM.from_pretrained(name_path, token=True,attn_implementation="eager", torch_dtype=torch_dtype)
    if only_tokenizer == False:
        model.to(device)
        model.zero_grad()
    else:
        model = None

    return model, tokenizer


def load_contributions(contributions_path):
    """
    returns a list of contribution matrices and their corresponding local and global ids.
    """
    with open(contributions_path, 'rb') as handle:
        o = pickle.load(handle)
    contr_list = [out['contributions_alti'] for out in o] # a list of samples with the contributions matrixes as elements; contr_matrix shape: [num_layers,gen_seq_length,input_seq_length] - input_seq_length is with target prefix
    ids_list = [out['id'] for out in o]
    input_tokens_list = [out['input_token_ids'] for out in o]
    gen_tokens_list = [out['gen_token_ids'] for out in o]
    return contr_list, ids_list,input_tokens_list,gen_tokens_list

def load_outputs(outputs_path):
    """
    returns two lists corresponding to input tokens and the generated tokens (for each sample)
    """
    with open(outputs_path, 'rb') as handle:
        o = pickle.load(handle)
    input_tokens_list = [out['input_token_ids'] for out in o] # a list of samples with the input tokens
    gen_tokens_list = [out['gen_token_ids'] for out in o] # a list of samples with the generated tokens
    return input_tokens_list, gen_tokens_list

def load_prep_data(data_path,model_version,keep_ids=None):
    """
    This functions loads the data and keeps the ids defined in keep_ids. It returns the needed data for the specific ids.
    The data are the input (prompt) and the generated sequence as strings.
    """
    with open(data_path,'rb') as f:
        data_list = json.load(f)
    data = [{"id":sample['index'],"prompt":sample[f'instruction-{model_version}'], "gen_seq":sample[f'{model_version}'] } for sample in data_list]

    prompts = [sample['prompt'].replace("\\n","\n") for sample in data if sample['id']  in keep_ids]
    gen_outs = [sample['gen_seq'].replace("\\n","\n") for sample in data if sample['id']  in keep_ids]
    return prompts,gen_outs

def get_language_name(lang_code):
    # Mapping of language codes to full language names
    language_mapping = {
        'en': 'English',
        'de': 'German',
        'fr': 'French',
        'es': 'Spanish',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'bn': 'Bengali',
        'tr': 'Turkish',
        'nl': 'Dutch',
        'sv': 'Swedish',
        'pl': 'Polish',
        'cs': 'Czech',
        'da': 'Danish',
        'fi': 'Finnish',
        'el': 'Greek',
        'he': 'Hebrew',
        'no': 'Norwegian',
        'hu': 'Hungarian',
        'ro': 'Romanian',
        'th': 'Thai',
        'vi': 'Vietnamese'
        # Add more as needed
    }
    return language_mapping[lang_code]

def regex_string_extraction(sequence,source_lang_code,target_lang_code):
    source_lang = get_language_name(source_lang_code)
    target_lang = get_language_name(target_lang_code)

    # Define patterns
    starting_pattern = r"(.*user\n)" #kepe group 0
    context_pattern = r"\n(Context:.*\n\n)Translate the" #keep group 1
    translate_pattern = fr"(Translate the.*{source_lang}.*{target_lang}(?::\n|, given the context.\n))" #keep group 0
    source_pattern = fr"Translate the.*{source_lang}.*{target_lang}(?::\n|, given the context.\n)(.*{source_lang}.*)<\|im_end\|>\n" #keep group 1
    end_pattern = r"<\|im_end\|>\n<\|im_start\|>assistant\n" #keep group 0

    # Initialize result dictionary
    result = {
        "start_prompt": None,
        "context": None,
        "instruction":None,
        "source": None,
        "end_prompt": None
    }

    # Extract starting prompt
    user_prompt_match = re.search(starting_pattern, sequence)
    if user_prompt_match:
        result["start_prompt"] = user_prompt_match.group(0).replace("\\n","\n")

    # Extract context
    context_match = re.search(context_pattern, sequence,re.DOTALL)
    if context_match:
        result["context"] = context_match.group(1).replace("\\n","\n")

    # Extract instruction
    translate_match = re.search(translate_pattern, sequence)
    if translate_match:
        result["instruction"] = translate_match.group(0).replace("\\n","\n")

    # Extract source
    source_match = re.search(source_pattern, sequence,re.DOTALL)
    if source_match:
        result["source"] = source_match.group(1).replace("\\n","\n")

    # Extract end pattern
    end_prompt_match = re.search(end_pattern, sequence)
    if end_prompt_match:
        result["end_prompt"] = end_prompt_match.group(0).replace("\\n","\n")
    return result


def get_prompt_tokenized_segments(sequence,tokenizer,source_lang_code,target_lang_code):
    # we split the prompt into the corresponding segments
    tokenized_res = {
        "start_prompt": None,
        "context": None,
        "instruction": None,
        "source": None,
        "end_prompt": None
    }
    results = regex_string_extraction(sequence,source_lang_code,target_lang_code)
    if results["start_prompt"] is not None:
        tokenized_res["start_prompt"] = tokenizer(results["start_prompt"],add_special_tokens=True)['input_ids'] # we add only in this case the starting token!
    if results["context"] is not None:
        tokenized_res["context"] = tokenizer(results["context"],add_special_tokens=False)['input_ids']
    if results["instruction"] is not None:
        tokenized_res["instruction"] = tokenizer(results["instruction"],add_special_tokens=False)['input_ids']
    if results["source"] is not None:
        tokenized_res["source"] = tokenizer(results["source"],add_special_tokens=False)['input_ids']
    if results["end_prompt"] is not None:
        tokenized_res["end_prompt"] = tokenizer(results["end_prompt"],add_special_tokens=False)['input_ids']
    tokenized_res = {key:val for key,val in tokenized_res.items() if val is not None}
    return tokenized_res


def get_target_prefix_tokenized_segments(sequence,tokenizer):
    # we dont want to split the target prefix into multiple bins. we treat it as a single segment in this implementation
    tokenized_res = {}
    tokenized_res["target_prefix"] = tokenizer(sequence,add_special_tokens=False)['input_ids']
    return tokenized_res


def get_generated_tokenized_segments(sequence,tokenizer):
    # we dont want to split the generated into multiple bins. we treat it as a single segment in this implementation
    tokenized_res = {}
    tokenized_res["generated"] = tokenizer(sequence,add_special_tokens=False)['input_ids']
    return tokenized_res

#TODO: segment lengths can be computed for each model over the tokens in different custom functions.

def find_all_sublists(lst, sublist):
    n = len(sublist)
    indices = []
    for i in range(len(lst) - n + 1):
        if lst[i:i+n] == sublist:
            indices.append(i)  # Store the starting index
    return indices

def prompt_segment_lengths_tower_instr_w_context(input_tokens,tokenizer):
    tok_segments = {}
    ret_indices = [index for index, elem in enumerate(input_tokens) if elem == 13] #identify indices of \n
    # the first 1 "\n" construct the start prompt segment
    start_prompt_seg = input_tokens[:ret_indices[0]+1]

    rest_prompt_seg = input_tokens[ret_indices[0]+1:]
    end_ = tokenizer("<|im_end|>\n",add_special_tokens=False)['input_ids'] # we will search for this to see where it appears
    ending_index = find_all_sublists(list(rest_prompt_seg),list(end_))[-1] #we get the latest index
    end_prompt_seg = rest_prompt_seg[ending_index:]
    rest_prompt_seg = rest_prompt_seg[:ending_index]

    #now we need to find the context, the translation and the source
    ret_indices = [index for index, elem in enumerate(rest_prompt_seg) if elem == 13]  # identify indices of \n in the rest_prompt
    source_seg = rest_prompt_seg[ret_indices[-2]+1:]
    rest_prompt_seg = rest_prompt_seg[:ret_indices[-2]+1]
    #now we need to find the context, the instruction and the source
    # we identify the 2nd \n from the end
    ret_indices = [index for index, elem in enumerate(rest_prompt_seg) if elem == 13]  # identify indices of \n in the rest_prompt
    instruction_seg = rest_prompt_seg[ret_indices[-2]+1:]
    context_seg = rest_prompt_seg[:ret_indices[-2] + 1]

    tok_segments["start_prompt"] = start_prompt_seg
    tok_segments["context"] = context_seg
    tok_segments["instruction"] = instruction_seg
    tok_segments["source"] = source_seg
    tok_segments["end_prompt"] = end_prompt_seg
    # compute segment lengths for prompt sequence
    prompt_segment_lengths = [len(seg) for seg_name, seg in tok_segments.items()]
    return {"segments":tok_segments,"lengths":prompt_segment_lengths}

def prompt_segment_lengths_tower_instr_wo_context(input_tokens,tokenizer):
    tok_segments = {}
    ret_indices = [index for index, elem in enumerate(input_tokens) if elem == 13]  # identify indices of \n
    # the first  "\n" construct the start prompt segment
    start_prompt_seg = input_tokens[:ret_indices[0] + 1]

    rest_prompt_seg = input_tokens[ret_indices[0] + 1:]
    end_ = tokenizer("<|im_end|>\n", add_special_tokens=False)[
        'input_ids']  # we will search for this to see where it appears
    ending_index = find_all_sublists(list(rest_prompt_seg), list(end_))[-1]  # we get the latest index
    end_prompt_seg = rest_prompt_seg[ending_index:]
    rest_prompt_seg = rest_prompt_seg[:ending_index]

    # now we need to find the context, the translation and the source
    ret_indices = [index for index, elem in enumerate(rest_prompt_seg) if
                   elem == 13]  # identify indices of \n in the rest_prompt
    source_seg = rest_prompt_seg[ret_indices[-2] + 1:]
    instruction_seg = rest_prompt_seg[:ret_indices[-2] + 1]

    tok_segments["start_prompt"] = start_prompt_seg
    tok_segments["instruction"] = instruction_seg
    tok_segments["source"] = source_seg
    tok_segments["end_prompt"] = end_prompt_seg
    # compute segment lengths for prompt sequence
    prompt_segment_lengths = [len(seg) for seg_name, seg in tok_segments.items()]
    return {"segments": tok_segments, "lengths": prompt_segment_lengths}

def prompt_segment_lengths_tower_chat_w_context(input_tokens,tokenizer):
    tok_segments = {}
    ret_indices = [index for index, elem in enumerate(input_tokens) if elem == 13] #identify indices of \n
    # the first 3 "\n" construct the start prompt segment
    start_prompt_seg = input_tokens[:ret_indices[2]+1]

    rest_prompt_seg = input_tokens[ret_indices[2]+1:]
    end_ = tokenizer("<|im_end|>\n",add_special_tokens=False)['input_ids'] # we will search for this to see where it appears
    ending_index = find_all_sublists(list(rest_prompt_seg),list(end_))[-1] #we get the latest index
    end_prompt_seg = rest_prompt_seg[ending_index:]
    rest_prompt_seg = rest_prompt_seg[:ending_index]

    #now we need to find the context, the translation and the source
    ret_indices = [index for index, elem in enumerate(rest_prompt_seg) if elem == 13]  # identify indices of \n in the rest_prompt
    source_seg = rest_prompt_seg[ret_indices[-2]+1:]
    rest_prompt_seg = rest_prompt_seg[:ret_indices[-2]+1]

    #now we need to find the context, the instruction and the source
    # we identify the 2nd \n from the end
    ret_indices = [index for index, elem in enumerate(rest_prompt_seg) if elem == 13]  # identify indices of \n in the rest_prompt
    instruction_seg = rest_prompt_seg[ret_indices[-2]+1:]
    context_seg = rest_prompt_seg[:ret_indices[-2] + 1]

    tok_segments["start_prompt"] = start_prompt_seg
    tok_segments["context"] = context_seg
    tok_segments["instruction"] = instruction_seg
    tok_segments["source"] = source_seg
    tok_segments["end_prompt"] = end_prompt_seg
    # compute segment lengths for prompt sequence
    prompt_segment_lengths = [len(seg) for seg_name, seg in tok_segments.items()]
    return {"segments":tok_segments,"lengths":prompt_segment_lengths}


def prompt_segment_lengths_tower_chat_wo_context(input_tokens,tokenizer):
    tok_segments = {}
    ret_indices = [index for index, elem in enumerate(input_tokens) if elem == 13] #identify indices of \n
    # the first 3 "\n" construct the start prompt segment
    start_prompt_seg = input_tokens[:ret_indices[2]+1]

    rest_prompt_seg = input_tokens[ret_indices[2]+1:]
    end_ = tokenizer("<|im_end|>\n",add_special_tokens=False)['input_ids'] # we will search for this to see where it appears
    ending_index = find_all_sublists(list(rest_prompt_seg),list(end_))[-1] #we get the latest index
    end_prompt_seg = rest_prompt_seg[ending_index:]
    rest_prompt_seg = rest_prompt_seg[:ending_index]

    #now we need to find the context, the translation and the source
    ret_indices = [index for index, elem in enumerate(rest_prompt_seg) if elem == 13]  # identify indices of \n in the rest_prompt
    source_seg = rest_prompt_seg[ret_indices[-2]+1:]
    instruction_seg = rest_prompt_seg[:ret_indices[-2]+1]

    tok_segments["start_prompt"] = start_prompt_seg
    tok_segments["instruction"] = instruction_seg
    tok_segments["source"] = source_seg
    tok_segments["end_prompt"] = end_prompt_seg
    # compute segment lengths for prompt sequence
    prompt_segment_lengths = [len(seg) for seg_name, seg in tok_segments.items()]
    return {"segments":tok_segments,"lengths":prompt_segment_lengths}

def tp_segment_lengths(tokens_list):
    """
    This function needs to be implemented by the user if we want to split the tp into multiple bins/segments.
    """
    tok_segments = {}
    tok_segments["target_prefix"] = tokens_list
    tp_segment_lengths = [len(seg) for seg_name, seg in tok_segments.items()]
    return {"segments": tok_segments, "lengths": tp_segment_lengths}

def gen_seq_segment_lengths(tokens_list):
    """
    This function needs to be implemented by the user if we want to split the generated sequence into multiple bins/segments.
    """
    tok_segments = {}
    tok_segments["gen"] = tokens_list
    tp_segment_lengths = [len(seg) for seg_name, seg in tok_segments.items()]
    return {"segments": tok_segments, "lengths": tp_segment_lengths}

def get_prompt_segments(model_version, tokenizer, prompt_tokens_all):
    if model_version == "greedy-7b-ft-w-context":
        segmentation_function = prompt_segment_lengths_tower_chat_w_context
    elif model_version == "greedy-7b-ft-wo-context":
        segmentation_function = prompt_segment_lengths_tower_chat_wo_context
    elif model_version=="greedy-7b-w-context":
        segmentation_function = prompt_segment_lengths_tower_instr_w_context
    elif model_version == "greedy-7b-wo-context":
        segmentation_function = prompt_segment_lengths_tower_instr_wo_context
    else:
        raise NotImplementedError
    prompt_segments_all = []
    #iterate over all samples and get prompt segments
    for index,input_tokens in enumerate(prompt_tokens_all):
        prompt_seg_dict = segmentation_function(input_tokens, tokenizer)
        prompt_segments_all.append(prompt_seg_dict)
    return prompt_segments_all

def get_target_prefix_segments(tokens_list_all):
    segmentation_function = tp_segment_lengths
    target_prefix_segments_all = []
    # we iterate over samples and do segmentations
    for target_prefix_tokens in tokens_list_all:
        tp_segment_dict = segmentation_function(target_prefix_tokens)
        target_prefix_segments_all.append(tp_segment_dict)
    return target_prefix_segments_all

def get_gen_seq_segments(tokens_list_all):
    segmentation_function = gen_seq_segment_lengths
    gen_seq_segments_all = []
    # we iterate over samples and do segmentations
    for gen_seq_tokens in tokens_list_all:
        gen_seq_segment_dict = segmentation_function(gen_seq_tokens)
        gen_seq_segments_all.append(gen_seq_segment_dict)
    return gen_seq_segments_all


def segment_lengths_computation(input_seq,gen_seq,tokenizer,source_lang_code,target_lang_code):
    """
    This function needs to be implemented/overwritten by the user.
    It should return 3 dicts
    The 1st defines the segments and the segment lengths of the prompt(w/o the target prefix) -- the lengths is a list
    The 2nd defines the segments and the segment lengths of the target prefix  -- the lengths is a list
    The 3rd defines the segments and the segment lengths of the generated seq  -- the lengths is a list
    """
    target_prefix = gen_seq[:-1]
    # compute segment lengths for prompt sequence
    prompt_segments = get_prompt_tokenized_segments(input_seq,tokenizer,source_lang_code,target_lang_code)
    prompt_segment_lengths = [len(seg) for seg_name, seg in prompt_segments.items()]

    # compute segment lengths for target prefix sequence and gen_seq
    tp_segments = get_target_prefix_tokenized_segments(target_prefix,tokenizer)
    tp_segment_lengths = [len(seg) for seg_name, seg in tp_segments.items()]

    # compute segment lengths for generated sequence
    gen_segments = get_generated_tokenized_segments(gen_seq,tokenizer)
    gen_segment_lengths = [len(seg) for seg_name, seg in gen_segments.items()]

    return {"segments": prompt_segments, "lengths": prompt_segment_lengths},{"segments": tp_segments, "lengths": tp_segment_lengths},{"segments": gen_segments, "lengths": gen_segment_lengths}


def aggregate_contributions_step(contr_matrix,input_prompt_tokens,gen_tokens,
                                 input_prompt_aggr_seg_lengths=None,tp_aggr_seg_lengths=None,
                                 gen_aggr_seg_lengths=None):
    """
    :param contr_matrix: contribution matrix for specific sample ,shape: [num_layers,gen_seq_length,input_seq_length] (input_seq length contains the target prefix)
    :param input_prompt_tokens: tokens of input prompt (w/o the target prefix) ,shape: [input_seq_length]
    :param gen_tokens: tokens of generated sequence (the target prefix is consider as gen_tokens[:-1]) , ,shape: [gen_seq_length]
    :param input_prompt_aggr_seg_lengths: segment lengths of input prompt (w/o the tp) to perform aggregation on
    :param tp_aggr_seg_lengths: segment lengths of target prefix to perform aggregation on
    :param gen_aggr_seg_lengths: segment lengths of generated seq (y-axis) to perform aggregation on
    :return: returns the aggregated contribution matrix
    """
    target_prefix_tokens=gen_tokens[:-1]

    # initially we check that all lengths are correctly given and do not exceed the tokenized sequences
    assert sum(input_prompt_aggr_seg_lengths)==len(input_prompt_tokens),"Incorrect segmentation in input prompt tokens.Given segments length are less or more than the length of the prompt."
    assert sum(tp_aggr_seg_lengths) == len(target_prefix_tokens), "Incorrect segmentation in target prefix tokens.Given segments length are less or more than the length of the target prefix."
    assert sum(gen_aggr_seg_lengths) == len(gen_tokens), "Incorrect segmentation in generated tokens(y-axis).Given segments length are less or more than the length of the generated tokens(y-axis)."

    assert contr_matrix.shape[1]==len(gen_tokens) and contr_matrix.shape[-1]==len(input_prompt_tokens)+len(target_prefix_tokens),"Contribution matrix shape does not match with the given sequences."
    # then we perform aggregation..
    index=0
    aggregation_contributions_list =[]

    #aggregation over input prompt
    for seg_length in input_prompt_aggr_seg_lengths:
        segment_contrs = contr_matrix[:, :, index:index + seg_length]
        aggr_contrs = torch.sum(segment_contrs, dim=-1)
        aggregation_contributions_list.append(aggr_contrs)
        index += seg_length
    prompt_aggr_contr = torch.stack(aggregation_contributions_list, dim=0).permute(1,2,0)

    #aggregation over the target prefix
    original_target_prefix_contr = contr_matrix[:, :, index:]
    target_prefix_contr = torch.zeros(contr_matrix.shape[0], contr_matrix.shape[1], len(tp_aggr_seg_lengths)) #we create a zero matrix with the number of tp segments at the last dim
    for row in range(1, original_target_prefix_contr.shape[1]):  # row==0 is skipped as it's always zero!
        active_target_prefix_contr = original_target_prefix_contr[:, row,
                                     range(0, row)]  # we have to aggregate only over the ones that are active!
        index = 0
        for bin_i, segment_len in enumerate(tp_aggr_seg_lengths):
            if segment_len >= active_target_prefix_contr.shape[-1] - index:
                tmp_contr = active_target_prefix_contr[:, range(index, active_target_prefix_contr.shape[-1])]  # from the "active" contributions we get those that fit the bin and are active!!
                aggr_contr = torch.sum(tmp_contr, dim=-1)  # /(active_target_prefix_contr.shape[-1]-index)
                target_prefix_contr[:, row, bin_i] = aggr_contr
                break
            else:
                # now we take only the contributions that fit inside the bin from the active ones and then move to the next bin!
                tmp_contr = active_target_prefix_contr[:, range(index, index + segment_len)]
                aggr_contr = torch.sum(tmp_contr, dim=-1)  # /segment_len
                target_prefix_contr[:, row, bin_i] = aggr_contr
                index += segment_len  # then we increase the index to point to the next elem that is does not fit in the bin

    input_aggr_contr = torch.cat((prompt_aggr_contr, target_prefix_contr), dim=-1)

    # we perform aggregation over the decoding step (y-axis).
    # In that case we average the contributions of each bin
    new_contr_matrix = []
    index = 0
    for bin_ind, segment_len in enumerate(gen_aggr_seg_lengths):
        tmp_aggr_contr = input_aggr_contr[:, range(index, index + segment_len)]
        new_contr_matrix.append(torch.mean(tmp_aggr_contr, dim=1))
        index += segment_len
    new_contr_matrix = torch.stack(new_contr_matrix, dim=0).permute(1, 0, 2)
    return new_contr_matrix


def aggregate_contributions(ids_list,contributions_list,input_tokens_list,gen_tokens_list,prompt_seg_lengths,
                            target_prefix_seg_lengths,gen_seq_seg_lengths):
    """
    """
    return_list = []
    for index, contr in enumerate(contributions_list):
        id = ids_list[index]

        aggr_contr_matrix = aggregate_contributions_step(contr_matrix=contributions_list[index],
                                     input_prompt_tokens=input_tokens_list[index],
                                     gen_tokens=gen_tokens_list[index],
                                     input_prompt_aggr_seg_lengths=prompt_seg_lengths[index],
                                     tp_aggr_seg_lengths=target_prefix_seg_lengths[index],
                                     gen_aggr_seg_lengths=gen_seq_seg_lengths[index])
        return_list.append({"id":id,"aggregated_contributions":aggr_contr_matrix})
    return return_list


def aggregate_contributions_normalized(ids_list,contributions_list,input_tokens_list,gen_tokens_list,prompt_seg_lengths,
                            target_prefix_seg_lengths,gen_seq_seg_lengths):
    """
    """
    return_list = []
    for index, contr in enumerate(contributions_list):
        id = ids_list[index]

        aggr_contr_matrix = aggregate_contributions_step_normalize(contr_matrix=contributions_list[index],
                                     input_prompt_tokens=input_tokens_list[index],
                                     gen_tokens=gen_tokens_list[index],
                                     input_prompt_aggr_seg_lengths=prompt_seg_lengths[index],
                                     tp_aggr_seg_lengths=target_prefix_seg_lengths[index],
                                     gen_aggr_seg_lengths=gen_seq_seg_lengths[index])
        return_list.append({"id":id,"aggregated_contributions":aggr_contr_matrix})
    return return_list


def aggregate_contributions_step_normalize(contr_matrix,input_prompt_tokens,gen_tokens,
                                 input_prompt_aggr_seg_lengths=None,tp_aggr_seg_lengths=None,
                                 gen_aggr_seg_lengths=None):
    """
    :param contr_matrix: contribution matrix for specific sample ,shape: [num_layers,gen_seq_length,input_seq_length] (input_seq length contains the target prefix)
    :param input_prompt_tokens: tokens of input prompt (w/o the target prefix) ,shape: [input_seq_length]
    :param gen_tokens: tokens of generated sequence (the target prefix is consider as gen_tokens[:-1]) , ,shape: [gen_seq_length]
    :param input_prompt_aggr_seg_lengths: segment lengths of input prompt (w/o the tp) to perform aggregation on
    :param tp_aggr_seg_lengths: segment lengths of target prefix to perform aggregation on
    :param gen_aggr_seg_lengths: segment lengths of generated seq (y-axis) to perform aggregation on
    :return: returns the aggregated contribution matrix
    """
    target_prefix_tokens=gen_tokens[:-1]

    # initially we check that all lengths are correctly given and do not exceed the tokenized sequences
    assert sum(input_prompt_aggr_seg_lengths)==len(input_prompt_tokens),"Incorrect segmentation in input prompt tokens.Given segments length are less or more than the length of the prompt."
    assert sum(tp_aggr_seg_lengths) == len(target_prefix_tokens), "Incorrect segmentation in target prefix tokens.Given segments length are less or more than the length of the target prefix."
    assert sum(gen_aggr_seg_lengths) == len(gen_tokens), "Incorrect segmentation in generated tokens(y-axis).Given segments length are less or more than the length of the generated tokens(y-axis)."

    assert contr_matrix.shape[1]==len(gen_tokens) and contr_matrix.shape[-1]==len(input_prompt_tokens)+len(target_prefix_tokens),"Contribution matrix shape does not match with the given sequences."
    # then we perform aggregation..
    index=0
    aggregation_contributions_list =[]

    #aggregation over input prompt
    for seg_length in input_prompt_aggr_seg_lengths:
        segment_contrs = contr_matrix[:, :, index:index + seg_length]
        aggr_contrs = torch.sum(segment_contrs, dim=-1) / seg_length
        aggregation_contributions_list.append(aggr_contrs)
        index += seg_length
    prompt_aggr_contr = torch.stack(aggregation_contributions_list, dim=0).permute(1,2,0)

    #aggregation over the target prefix
    original_target_prefix_contr = contr_matrix[:, :, index:]
    target_prefix_contr = torch.zeros(contr_matrix.shape[0], contr_matrix.shape[1], len(tp_aggr_seg_lengths)) #we create a zero matrix with the number of tp segments at the last dim
    for row in range(1, original_target_prefix_contr.shape[1]):  # row==0 is skipped as it's always zero!
        active_target_prefix_contr = original_target_prefix_contr[:, row,
                                     range(0, row)]  # we have to aggregate only over the ones that are active!
        index = 0
        for bin_i, segment_len in enumerate(tp_aggr_seg_lengths):
            if segment_len >= active_target_prefix_contr.shape[-1] - index:
                tmp_contr = active_target_prefix_contr[:, range(index, active_target_prefix_contr.shape[-1])]  # from the "active" contributions we get those that fit the bin and are active!!
                aggr_contr = torch.sum(tmp_contr, dim=-1)  /(active_target_prefix_contr.shape[-1]-index)
                target_prefix_contr[:, row, bin_i] = aggr_contr
                break
            else:
                # now we take only the contributions that fit inside the bin from the active ones and then move to the next bin!
                tmp_contr = active_target_prefix_contr[:, range(index, index + segment_len)]
                aggr_contr = torch.sum(tmp_contr, dim=-1)  /segment_len
                target_prefix_contr[:, row, bin_i] = aggr_contr
                index += segment_len  # then we increase the index to point to the next elem that is does not fit in the bin

    input_aggr_contr = torch.cat((prompt_aggr_contr, target_prefix_contr), dim=-1)
    norm = torch.sum(input_aggr_contr, dim=-1).unsqueeze(-1)
    input_aggr_contr = input_aggr_contr / norm


    # we perform aggregation over the decoding step (y-axis).
    # In that case we average the contributions of each bin
    new_contr_matrix = []
    index = 0
    for bin_ind, segment_len in enumerate(gen_aggr_seg_lengths):
        tmp_aggr_contr = input_aggr_contr[:, range(index, index + segment_len)]
        new_contr_matrix.append(torch.mean(tmp_aggr_contr, dim=1))
        index += segment_len
    new_contr_matrix = torch.stack(new_contr_matrix, dim=0).permute(1, 0, 2)
    return new_contr_matrix



class SegmentationClass:
    def __init__(self,context_type_segmentation=None,model_version=None,model_tokenizer=None,num_turns=None):
        if context_type_segmentation not in ['turn-level','context-level','no-context']:
            assert False, 'Invalid context type segmentation'
        if model_version not in ['TowerInstruct-7B-w-chat-empty-sys','TowerInstruct-7B-v0.2']:
            assert False, 'Invalid model version!'
        self.context_type_segmentation = context_type_segmentation
        self.model_ver = model_version
        self.tokenizer = model_tokenizer

        if self.context_type_segmentation == 'turn-level':
            if num_turns is None:
                assert False, ('You need to specify number of turns for turn-level segmentation. This is required '
                               'for validating that the number of turns in the context is correct and the same for all'
                               'samples.')
            self.num_turns = num_turns
            if self.model_ver == 'TowerInstruct-7B-w-chat-empty-sys':
                self.segmentation_function = self.prompt_segment_lengths_tower_chat_turn_level
            elif self.model_ver == 'TowerInstruct-7B-v0.2':
                self.segmentation_function = self.prompt_segment_lengths_tower_instr_turn_level
        elif self.context_type_segmentation == 'context-level':
            if self.model_ver == 'TowerInstruct-7B-w-chat-empty-sys':
                self.segmentation_function = self.prompt_segment_lengths_tower_chat_w_context
            elif self.model_ver == 'TowerInstruct-7B-v0.2':
                self.segmentation_function = self.prompt_segment_lengths_tower_instr_w_context
        elif self.context_type_segmentation == 'no-context':
            if self.model_ver == 'TowerInstruct-7B-w-chat-empty-sys':
                self.segmentation_function = self.prompt_segment_lengths_tower_chat_wo_context
            elif self.model_ver == 'TowerInstruct-7B-v0.2':
                self.segmentation_function = self.prompt_segment_lengths_tower_instr_wo_context
        else:
            assert False, 'Invalid context type segmentation!'

    def prompt_segment_lengths_tower_chat_w_context(self,input_tokens):
        tok_segments = {}
        ret_indices = [index for index, elem in enumerate(input_tokens) if elem == 13]  # identify indices of \n
        # the first 3 "\n" construct the start prompt segment
        start_prompt_seg = input_tokens[:ret_indices[2] + 1]

        rest_prompt_seg = input_tokens[ret_indices[2] + 1:]
        end_ = self.tokenizer("<|im_end|>\n", add_special_tokens=False)[
            'input_ids']  # we will search for this to see where it appears
        ending_index = find_all_sublists(list(rest_prompt_seg), list(end_))[-1]  # we get the latest index
        end_prompt_seg = rest_prompt_seg[ending_index:]
        rest_prompt_seg = rest_prompt_seg[:ending_index]

        # now we need to find the context, the translation and the source
        ret_indices = [index for index, elem in enumerate(rest_prompt_seg) if
                       elem == 13]  # identify indices of \n in the rest_prompt
        source_seg = rest_prompt_seg[ret_indices[-2] + 1:]
        rest_prompt_seg = rest_prompt_seg[:ret_indices[-2] + 1]

        # now we need to find the context, the instruction and the source
        # we identify the 2nd \n from the end
        ret_indices = [index for index, elem in enumerate(rest_prompt_seg) if
                       elem == 13]  # identify indices of \n in the rest_prompt
        instruction_seg = rest_prompt_seg[ret_indices[-2] + 1:]
        context_seg = rest_prompt_seg[:ret_indices[-2] + 1]

        tok_segments["start_prompt"] = start_prompt_seg
        tok_segments["context"] = context_seg
        tok_segments["instruction"] = instruction_seg
        tok_segments["source"] = source_seg
        tok_segments["end_prompt"] = end_prompt_seg
        # compute segment lengths for prompt sequence
        prompt_segment_lengths = [len(seg) for seg_name, seg in tok_segments.items()]
        return {"segments": tok_segments, "lengths": prompt_segment_lengths}

    def prompt_segment_lengths_tower_chat_wo_context(self,input_tokens,):
        tok_segments = {}
        ret_indices = [index for index, elem in enumerate(input_tokens) if elem == 13]  # identify indices of \n
        # the first 3 "\n" construct the start prompt segment
        start_prompt_seg = input_tokens[:ret_indices[2] + 1]

        rest_prompt_seg = input_tokens[ret_indices[2] + 1:]
        end_ = self.tokenizer("<|im_end|>\n", add_special_tokens=False)[
            'input_ids']  # we will search for this to see where it appears
        ending_index = find_all_sublists(list(rest_prompt_seg), list(end_))[-1]  # we get the latest index
        end_prompt_seg = rest_prompt_seg[ending_index:]
        rest_prompt_seg = rest_prompt_seg[:ending_index]

        # now we need to find the context, the translation and the source
        ret_indices = [index for index, elem in enumerate(rest_prompt_seg) if
                       elem == 13]  # identify indices of \n in the rest_prompt
        source_seg = rest_prompt_seg[ret_indices[-2] + 1:]
        instruction_seg = rest_prompt_seg[:ret_indices[-2] + 1]

        tok_segments["start_prompt"] = start_prompt_seg
        tok_segments["instruction"] = instruction_seg
        tok_segments["source"] = source_seg
        tok_segments["end_prompt"] = end_prompt_seg
        # compute segment lengths for prompt sequence
        prompt_segment_lengths = [len(seg) for seg_name, seg in tok_segments.items()]
        return {"segments": tok_segments, "lengths": prompt_segment_lengths}

    def prompt_segment_lengths_tower_instr_w_context(self,input_tokens):
        tok_segments = {}
        ret_indices = [index for index, elem in enumerate(input_tokens) if elem == 13] #identify indices of \n
        # the first 1 "\n" construct the start prompt segment
        start_prompt_seg = input_tokens[:ret_indices[0]+1]

        rest_prompt_seg = input_tokens[ret_indices[0]+1:]
        end_ = self.tokenizer("<|im_end|>\n",add_special_tokens=False)['input_ids'] # we will search for this to see where it appears
        ending_index = find_all_sublists(list(rest_prompt_seg),list(end_))[-1] #we get the latest index
        end_prompt_seg = rest_prompt_seg[ending_index:]
        rest_prompt_seg = rest_prompt_seg[:ending_index]

        #now we need to find the context, the translation and the source
        ret_indices = [index for index, elem in enumerate(rest_prompt_seg) if elem == 13]  # identify indices of \n in the rest_prompt
        source_seg = rest_prompt_seg[ret_indices[-2]+1:]
        rest_prompt_seg = rest_prompt_seg[:ret_indices[-2]+1]
        #now we need to find the context, the instruction and the source
        # we identify the 2nd \n from the end
        ret_indices = [index for index, elem in enumerate(rest_prompt_seg) if elem == 13]  # identify indices of \n in the rest_prompt
        instruction_seg = rest_prompt_seg[ret_indices[-2]+1:]
        context_seg = rest_prompt_seg[:ret_indices[-2] + 1]

        tok_segments["start_prompt"] = start_prompt_seg
        tok_segments["context"] = context_seg
        tok_segments["instruction"] = instruction_seg
        tok_segments["source"] = source_seg
        tok_segments["end_prompt"] = end_prompt_seg
        # compute segment lengths for prompt sequence
        prompt_segment_lengths = [len(seg) for seg_name, seg in tok_segments.items()]
        return {"segments":tok_segments,"lengths":prompt_segment_lengths}

    def prompt_segment_lengths_tower_instr_wo_context(self,input_tokens):
        tok_segments = {}
        ret_indices = [index for index, elem in enumerate(input_tokens) if elem == 13]  # identify indices of \n
        # the first  "\n" construct the start prompt segment
        start_prompt_seg = input_tokens[:ret_indices[0] + 1]

        rest_prompt_seg = input_tokens[ret_indices[0] + 1:]
        end_ = self.tokenizer("<|im_end|>\n", add_special_tokens=False)[
            'input_ids']  # we will search for this to see where it appears
        ending_index = find_all_sublists(list(rest_prompt_seg), list(end_))[-1]  # we get the latest index
        end_prompt_seg = rest_prompt_seg[ending_index:]
        rest_prompt_seg = rest_prompt_seg[:ending_index]

        # now we need to find the context, the translation and the source
        ret_indices = [index for index, elem in enumerate(rest_prompt_seg) if
                       elem == 13]  # identify indices of \n in the rest_prompt
        source_seg = rest_prompt_seg[ret_indices[-2] + 1:]
        instruction_seg = rest_prompt_seg[:ret_indices[-2] + 1]

        tok_segments["start_prompt"] = start_prompt_seg
        tok_segments["instruction"] = instruction_seg
        tok_segments["source"] = source_seg
        tok_segments["end_prompt"] = end_prompt_seg
        # compute segment lengths for prompt sequence
        prompt_segment_lengths = [len(seg) for seg_name, seg in tok_segments.items()]
        return {"segments": tok_segments, "lengths": prompt_segment_lengths}

    def prompt_segment_lengths_tower_chat_turn_level(self,input_tokens):
        tok_segments = {}
        ret_indices = [index for index, elem in enumerate(input_tokens) if elem == 13]  # identify indices of \n
        # the first 3 "\n" construct the start prompt segment
        start_prompt_seg = input_tokens[:ret_indices[2] + 1]

        rest_prompt_seg = input_tokens[ret_indices[2] + 1:]
        end_ = self.tokenizer("<|im_end|>\n", add_special_tokens=False)[
            'input_ids']  # we will search for this to see where it appears
        ending_index = find_all_sublists(list(rest_prompt_seg), list(end_))[-1]  # we get the latest index
        end_prompt_seg = rest_prompt_seg[ending_index:]
        rest_prompt_seg = rest_prompt_seg[:ending_index]

        # now we need to find the context, the translation and the source
        ret_indices = [index for index, elem in enumerate(rest_prompt_seg) if
                       elem == 13]  # identify indices of \n in the rest_prompt
        source_seg = rest_prompt_seg[ret_indices[-2] + 1:]
        rest_prompt_seg = rest_prompt_seg[:ret_indices[-2] + 1]

        # now we need to find the context, the instruction and the source
        # we identify the 2nd \n from the end
        ret_indices = [index for index, elem in enumerate(rest_prompt_seg) if
                       elem == 13]  # identify indices of \n in the rest_prompt
        instruction_seg = rest_prompt_seg[ret_indices[-2] + 1:]
        context_seg = rest_prompt_seg[:ret_indices[-2] + 1]

        # now that we have the context_segment we need to split it into turns and  verify that is the correct number!
        turns_indices = [i + 1 for i, elem in enumerate(context_seg[:-1]) if elem == 13]
        turn_segs = list(torch.tensor_split(context_seg, turns_indices))
        # we add the last \n to the 2nd  turn from the end and we remove the last turn (\n)
        turn_segs[-2] = torch.concat((turn_segs[-2],turn_segs[-1]))
        turn_segs = turn_segs[:-1]
        assert len(turn_segs) == self.num_turns,"Number of segmented turns does not match number of turns."

        tok_segments["start_prompt"] = start_prompt_seg
        for i,turn in enumerate(turn_segs):
            tok_segments[f"turn_{i+1}"] = turn
        tok_segments["instruction"] = instruction_seg
        tok_segments["source"] = source_seg
        tok_segments["end_prompt"] = end_prompt_seg
        # compute segment lengths for prompt sequence
        prompt_segment_lengths = [len(seg) for seg_name, seg in tok_segments.items()]
        return {"segments": tok_segments, "lengths": prompt_segment_lengths}

    def prompt_segment_lengths_tower_instr_turn_level(self,input_tokens):
        tok_segments = {}
        ret_indices = [index for index, elem in enumerate(input_tokens) if elem == 13] #identify indices of \n
        # the first 1 "\n" construct the start prompt segment
        start_prompt_seg = input_tokens[:ret_indices[0]+1]

        rest_prompt_seg = input_tokens[ret_indices[0]+1:]
        end_ = self.tokenizer("<|im_end|>\n",add_special_tokens=False)['input_ids'] # we will search for this to see where it appears
        ending_index = find_all_sublists(list(rest_prompt_seg),list(end_))[-1] #we get the latest index
        end_prompt_seg = rest_prompt_seg[ending_index:]
        rest_prompt_seg = rest_prompt_seg[:ending_index]

        #now we need to find the context, the translation and the source
        ret_indices = [index for index, elem in enumerate(rest_prompt_seg) if elem == 13]  # identify indices of \n in the rest_prompt
        source_seg = rest_prompt_seg[ret_indices[-2]+1:]
        rest_prompt_seg = rest_prompt_seg[:ret_indices[-2]+1]
        #now we need to find the context, the instruction and the source
        # we identify the 2nd \n from the end
        ret_indices = [index for index, elem in enumerate(rest_prompt_seg) if elem == 13]  # identify indices of \n in the rest_prompt
        instruction_seg = rest_prompt_seg[ret_indices[-2]+1:]
        context_seg = rest_prompt_seg[:ret_indices[-2] + 1]

        # now that we have the context_segment we need to split it into turns and  verify that is the correct number!
        # now that we have the context_segment we need to split it into turns and  verify that is the correct number!
        turns_indices = [i + 1 for i, elem in enumerate(context_seg[:-1]) if elem == 13]
        turn_segs = list(torch.tensor_split(context_seg, turns_indices))
        # we add the last \n to the 2nd  turn from the end and we remove the last turn (\n)
        turn_segs[-2] = torch.concat((turn_segs[-2], turn_segs[-1]))
        turn_segs = turn_segs[:-1]
        assert len(turn_segs) == self.num_turns, "Number of segmented turns does not match number of turns."

        tok_segments["start_prompt"] = start_prompt_seg
        for i, turn in enumerate(turn_segs):
            tok_segments[f"turn_{i + 1}"] = turn
        tok_segments["instruction"] = instruction_seg
        tok_segments["source"] = source_seg
        tok_segments["end_prompt"] = end_prompt_seg
        # compute segment lengths for prompt sequence
        prompt_segment_lengths = [len(seg) for seg_name, seg in tok_segments.items()]
        return {"segments":tok_segments,"lengths":prompt_segment_lengths}

    def tp_segment_lengths(self,tokens_list):
        """
        This function needs to be implemented by the user if we want to split the target prefix into
         multiple bins/segments.
        """
        tok_segments = {}
        tok_segments["target_prefix"] = tokens_list
        tp_segment_lengths = [len(seg) for seg_name, seg in tok_segments.items()]
        return {"segments": tok_segments, "lengths": tp_segment_lengths}

    def gen_seq_segment_lengths(self,tokens_list):
        """
        This function needs to be implemented by the user if we want to split the generated sequence into
         multiple bins/segments.
        """
        tok_segments = {}
        tok_segments["gen"] = tokens_list
        tp_segment_lengths = [len(seg) for seg_name, seg in tok_segments.items()]
        return {"segments": tok_segments, "lengths": tp_segment_lengths}

    def get_prompt_segments(self, prompt_tokens_all):
        """
        This calls the segmentation function for each prompt in prompt_tokens_all list. It returns a
        list (samples) containing dicts. Each dict includes the "segments" and the "segment_lengths".
        :param prompt_tokens_all:  List with all prompts , each element is a tensor of tokens.
        :return: a list (samples) containing dicts of segments (start_prompt,context/turns,source, end_prompt ) and
        their corresponding lengths.
        """
        prompt_segments_all = []
        # iterate over all samples and get prompt segments
        for index, input_tokens in enumerate(prompt_tokens_all):
            prompt_seg_dict = self.segmentation_function(input_tokens)
            prompt_segments_all.append(prompt_seg_dict)
        return prompt_segments_all

    def get_target_prefix_segments(self,tokens_list_all):
        segmentation_function = tp_segment_lengths
        target_prefix_segments_all = []
        # we iterate over samples and do segmentations
        for target_prefix_tokens in tokens_list_all:
            tp_segment_dict = segmentation_function(target_prefix_tokens)
            target_prefix_segments_all.append(tp_segment_dict)
        return target_prefix_segments_all

    def get_gen_seq_segments(self,tokens_list_all):
        segmentation_function = gen_seq_segment_lengths
        gen_seq_segments_all = []
        # we iterate over samples and do segmentations
        for gen_seq_tokens in tokens_list_all:
            gen_seq_segment_dict = segmentation_function(gen_seq_tokens)
            gen_seq_segments_all.append(gen_seq_segment_dict)
        return gen_seq_segments_all


class ContributionsAggregatorClass:

    def __init__(self,normalized_contributions=False, segmentator=None):
        self.normalized_contributions = normalized_contributions
        self.segmentator = segmentator
        if self.normalized_contributions:
            self.aggregator_func = self.aggregate_contributions_step_normalize
        else:
            self.aggregator_func = self.aggregate_contributions_step


    def aggregate_contributions_step(self, contr_matrix, input_prompt_aggr_seg_lengths=None, tp_aggr_seg_lengths=None,
                                     gen_aggr_seg_lengths=None):
        """
        :param contr_matrix: contribution matrix for specific sample ,shape: [num_layers,gen_seq_length,input_seq_length] (input_seq length contains the target prefix)
        :param input_prompt_aggr_seg_lengths: segment lengths of input prompt (w/o the tp) to perform aggregation on
        :param tp_aggr_seg_lengths: segment lengths of target prefix to perform aggregation on
        :param gen_aggr_seg_lengths: segment lengths of generated seq (y-axis) to perform aggregation on
        :return: returns the aggregated contribution matrix
        """
        index = 0
        aggregation_contributions_list = []

        # aggregation over input prompt
        for seg_length in input_prompt_aggr_seg_lengths:
            segment_contrs = contr_matrix[:, :, index:index + seg_length]
            aggr_contrs = torch.sum(segment_contrs, dim=-1)
            aggregation_contributions_list.append(aggr_contrs)
            index += seg_length
        prompt_aggr_contr = torch.stack(aggregation_contributions_list, dim=0).permute(1, 2, 0)

        # aggregation over the target prefix
        original_target_prefix_contr = contr_matrix[:, :, index:]
        target_prefix_contr = torch.zeros(contr_matrix.shape[0], contr_matrix.shape[1],
                                          len(tp_aggr_seg_lengths))  # we create a zero matrix with the number of tp segments at the last dim
        for row in range(1, original_target_prefix_contr.shape[1]):  # row==0 is skipped as it's always zero!
            active_target_prefix_contr = original_target_prefix_contr[:, row,
                                         range(0, row)]  # we have to aggregate only over the ones that are active!
            index = 0
            for bin_i, segment_len in enumerate(tp_aggr_seg_lengths):
                if segment_len >= active_target_prefix_contr.shape[-1] - index:
                    tmp_contr = active_target_prefix_contr[:, range(index, active_target_prefix_contr.shape[
                        -1])]  # from the "active" contributions we get those that fit the bin and are active!!
                    aggr_contr = torch.sum(tmp_contr, dim=-1)  # /(active_target_prefix_contr.shape[-1]-index)
                    target_prefix_contr[:, row, bin_i] = aggr_contr
                    break
                else:
                    # now we take only the contributions that fit inside the bin from the active ones and then move to the next bin!
                    tmp_contr = active_target_prefix_contr[:, range(index, index + segment_len)]
                    aggr_contr = torch.sum(tmp_contr, dim=-1)  # /segment_len
                    target_prefix_contr[:, row, bin_i] = aggr_contr
                    index += segment_len  # then we increase the index to point to the next elem that is does not fit in the bin

        input_aggr_contr = torch.cat((prompt_aggr_contr, target_prefix_contr), dim=-1)

        # we perform aggregation over the decoding step (y-axis).
        # In that case we average the contributions of each bin
        new_contr_matrix = []
        index = 0
        for bin_ind, segment_len in enumerate(gen_aggr_seg_lengths):
            tmp_aggr_contr = input_aggr_contr[:, range(index, index + segment_len)]
            new_contr_matrix.append(torch.mean(tmp_aggr_contr, dim=1))
            index += segment_len
        new_contr_matrix = torch.stack(new_contr_matrix, dim=0).permute(1, 0, 2)
        return new_contr_matrix


    def aggregate_contributions_step_normalize(self,contr_matrix, input_prompt_aggr_seg_lengths=None,
                                               tp_aggr_seg_lengths=None,
                                               gen_aggr_seg_lengths=None):
        """
        :param contr_matrix: contribution matrix for specific sample ,shape: [num_layers,gen_seq_length,input_seq_length] (input_seq length contains the target prefix)
        :param input_prompt_aggr_seg_lengths: segment lengths of input prompt (w/o the tp) to perform aggregation on
        :param tp_aggr_seg_lengths: segment lengths of target prefix to perform aggregation on
        :param gen_aggr_seg_lengths: segment lengths of generated seq (y-axis) to perform aggregation on
        :return: returns the aggregated contribution matrix
        """
        index = 0
        aggregation_contributions_list = []

        # aggregation over input prompt
        for seg_length in input_prompt_aggr_seg_lengths:
            segment_contrs = contr_matrix[:, :, index:index + seg_length]
            aggr_contrs = torch.sum(segment_contrs, dim=-1) / seg_length
            aggregation_contributions_list.append(aggr_contrs)
            index += seg_length
        prompt_aggr_contr = torch.stack(aggregation_contributions_list, dim=0).permute(1, 2, 0)

        # aggregation over the target prefix
        original_target_prefix_contr = contr_matrix[:, :, index:]
        target_prefix_contr = torch.zeros(contr_matrix.shape[0], contr_matrix.shape[1],
                                          len(tp_aggr_seg_lengths))  # we create a zero matrix with the number of tp segments at the last dim
        for row in range(1, original_target_prefix_contr.shape[1]):  # row==0 is skipped as it's always zero!
            active_target_prefix_contr = original_target_prefix_contr[:, row,
                                         range(0, row)]  # we have to aggregate only over the ones that are active!
            index = 0
            for bin_i, segment_len in enumerate(tp_aggr_seg_lengths):
                if segment_len >= active_target_prefix_contr.shape[-1] - index:
                    tmp_contr = active_target_prefix_contr[:, range(index, active_target_prefix_contr.shape[
                        -1])]  # from the "active" contributions we get those that fit the bin and are active!!
                    aggr_contr = torch.sum(tmp_contr, dim=-1) / (active_target_prefix_contr.shape[-1] - index)
                    target_prefix_contr[:, row, bin_i] = aggr_contr
                    break
                else:
                    # now we take only the contributions that fit inside the bin from the active ones and then move to the next bin!
                    tmp_contr = active_target_prefix_contr[:, range(index, index + segment_len)]
                    aggr_contr = torch.sum(tmp_contr, dim=-1) / segment_len
                    target_prefix_contr[:, row, bin_i] = aggr_contr
                    index += segment_len  # then we increase the index to point to the next elem that is does not fit in the bin

        input_aggr_contr = torch.cat((prompt_aggr_contr, target_prefix_contr), dim=-1)
        norm = torch.sum(input_aggr_contr, dim=-1).unsqueeze(-1)
        input_aggr_contr = input_aggr_contr / norm

        # we perform aggregation over the decoding step (y-axis).
        # In that case we average the contributions of each bin
        new_contr_matrix = []
        index = 0
        for bin_ind, segment_len in enumerate(gen_aggr_seg_lengths):
            tmp_aggr_contr = input_aggr_contr[:, range(index, index + segment_len)]
            new_contr_matrix.append(torch.mean(tmp_aggr_contr, dim=1))
            index += segment_len
        new_contr_matrix = torch.stack(new_contr_matrix, dim=0).permute(1, 0, 2)
        return new_contr_matrix

    def aggregate_contributions(self,ids_list,contributions_list,prompt_seg_lengths,target_prefix_seg_lengths,
                                gen_seq_seg_lengths):
        aggr_contr_list = []
        for index, contr in enumerate(contributions_list):
            id = ids_list[index]
            sample_contr = contributions_list[index]
            sample_prompt_lengths = prompt_seg_lengths[index]
            sample_tp_lengths = target_prefix_seg_lengths[index]
            sample_gen_seq_lengths = gen_seq_seg_lengths[index]
            failed = self.sanity_check(sample_contr,sample_prompt_lengths,sample_tp_lengths,sample_gen_seq_lengths)
            if failed:
                assert False, f'Sanity check not passed in aggregate_contributions at id: {id}'
            aggr_contr_matrix = self.aggregator_func(contr_matrix=sample_contr,
                                                     input_prompt_aggr_seg_lengths=sample_prompt_lengths,
                                                     tp_aggr_seg_lengths=sample_tp_lengths,
                                                     gen_aggr_seg_lengths=sample_gen_seq_lengths)
            aggr_contr_list.append(aggr_contr_matrix)
        return aggr_contr_list

    def get_aggregated_contributions(self, ids_list,contrs_list,prompt_tokens_list, gen_seq_tokens_list):

        segmented_prompts = self.segmentator.get_prompt_segments(prompt_tokens_list)
        #for target prefix we exclude the last generated token!
        segmented_tps = self.segmentator.get_target_prefix_segments([elem[:-1] for elem in gen_seq_tokens_list])
        segmented_gens = self.segmentator.get_gen_seq_segments(gen_seq_tokens_list)

        # we get now the lengths of the corresponding segments (they are used in the aggregation)
        prompt_segments_all = [sample['segments'] for sample in segmented_prompts]
        prompt_segment_lengths_all = [sample['lengths'] for sample in segmented_prompts]
        tp_segments_all = [sample['segments'] for sample in segmented_tps]
        tp_segment_lengths_all = [sample['lengths'] for sample in segmented_tps]
        gen_segments_all = [sample['segments'] for sample in segmented_gens]
        gen_segment_lengths_all = [sample['lengths'] for sample in segmented_gens]

        aggr_contributions_list = self.aggregate_contributions(ids_list, contrs_list,prompt_segment_lengths_all,
                                                               tp_segment_lengths_all,
                                                               gen_segment_lengths_all)

        sample_segments_list = []  # we gather the corresponding segments and save them as well
        for index in range(len(aggr_contributions_list)):
            segments_dict = {key: prompt_segments_all[index][key].tolist() for key in prompt_segments_all[index]}
            segments_dict.update({key: tp_segments_all[index][key].tolist() for key in tp_segments_all[index]})
            segments_dict.update({key: gen_segments_all[index][key].tolist() for key in gen_segments_all[index]})
            sample_segments_list.append(segments_dict)

        return {"ids":ids_list, "segments":sample_segments_list,"aggregated_contrs":aggr_contributions_list}

    def sanity_check(self,sample_contr,prompt_seg_lengths,sample_tp_seg_lengths,sample_gen_seq_lengths):
        failed = False
        if sum(prompt_seg_lengths) + sum(sample_tp_seg_lengths) != sample_contr.shape[-1]:
            failed=True
        if sum(sample_gen_seq_lengths) != sample_contr.shape[1]:
            failed=True
        return failed