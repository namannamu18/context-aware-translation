# Smoke-test model -- NOT a research model

Tiny Llama-architecture LM trained from scratch for a few CPU-minutes on BMELD train prompts (scripts/smoke_models.py). Used only to check that the pipeline runs end-to-end. Its outputs/scores are meaningless.

{
  "config": {
    "vocab_size": 8000,
    "max_position_embeddings": 2048,
    "hidden_size": 256,
    "intermediate_size": 768,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "hidden_act": "silu",
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-06,
    "pretraining_tp": 1,
    "use_cache": true,
    "rope_theta": 10000.0,
    "rope_scaling": null,
    "attention_bias": false,
    "attention_dropout": 0.0,
    "mlp_bias": false,
    "head_dim": 64,
    "return_dict": true,
    "output_hidden_states": false,
    "output_attentions": false,
    "torchscript": false,
    "torch_dtype": "float32",
    "use_bfloat16": false,
    "tf_legacy_loss": false,
    "pruned_heads": {},
    "tie_word_embeddings": true,
    "chunk_size_feed_forward": 0,
    "is_encoder_decoder": false,
    "is_decoder": false,
    "cross_attention_hidden_size": null,
    "add_cross_attention": false,
    "tie_encoder_decoder": false,
    "max_length": 20,
    "min_length": 0,
    "do_sample": false,
    "early_stopping": false,
    "num_beams": 1,
    "num_beam_groups": 1,
    "diversity_penalty": 0.0,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
    "typical_p": 1.0,
    "repetition_penalty": 1.0,
    "length_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "encoder_no_repeat_ngram_size": 0,
    "bad_words_ids": null,
    "num_return_sequences": 1,
    "output_scores": false,
    "return_dict_in_generate": false,
    "forced_bos_token_id": null,
    "forced_eos_token_id": null,
    "remove_invalid_values": false,
    "exponential_decay_length_penalty": null,
    "suppress_tokens": null,
    "begin_suppress_tokens": null,
    "architectures": [
      "LlamaForCausalLM"
    ],
    "finetuning_task": null,
    "id2label": {
      "0": "LABEL_0",
      "1": "LABEL_1"
    },
    "label2id": {
      "LABEL_0": 0,
      "LABEL_1": 1
    },
    "tokenizer_class": null,
    "prefix": null,
    "bos_token_id": 0,
    "pad_token_id": 2,
    "eos_token_id": 4,
    "sep_token_id": null,
    "decoder_start_token_id": null,
    "task_specific_params": null,
    "problem_type": null,
    "_name_or_path": "<scratch>/smoke_models/tiny-chatml-lm-stage1",
    "_attn_implementation_autoset": false,
    "transformers_version": "4.49.0",
    "model_type": "llama"
  },
  "train_log": [
    {
      "step": 1900,
      "loss": 2.2721,
      "minutes": 31.95
    },
    {
      "step": 1950,
      "loss": 1.9372,
      "minutes": 32.67
    },
    {
      "step": 2000,
      "loss": 2.3641,
      "minutes": 33.4
    },
    {
      "step": 2050,
      "loss": 2.0414,
      "minutes": 34.12
    },
    {
      "step": 2100,
      "loss": 2.2752,
      "minutes": 34.87
    }
  ]
}