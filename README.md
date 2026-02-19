## Overview

This repository contains the data and code to reproduce the results of our paper ["A Context-aware Framework for Translation-mediated Conversations"](https://arxiv.org/abs/2412.04205), of our [Unbabel+IT 2024 chat shared task submission](https://aclanthology.org/2024.wmt-1.100.pdf).

## Installation

Install [`tower-eval`](https://github.com/deep-spin/tower-eval) to run the generation and evaluation configs.


## Folders

- **``candidates``**: candidates sampled to perform MBR on.
  
- **``configs``**: tower-eval config to run generations.

- **``downloaded_data``**: raw data downloaded from shared task and MAIA github repositories.

- **``evaluations``**: automatic metric scores for generated translations as specified by tower-eval config file.

- **``generations``**: generated output folder containing translations by tower-eval config file.

-  **``instructions``**:  prompts for generating translations, used by tower-eval.

-  **``mbr_outputs``**: one best pick outputs using different comet and context-comet variants

-  **``notebooks``**: notebooks for preparing data and analyzing results.

-  **``paper_results``**: dataframes containing all translation outputs and comet scores for dev/test split as well as muda json files.

-  **``pcxmi``**: logprobs for towerinstruct and towerchat models. 

-  **``plots``**: plots included in the paper submissions.

-  **``instructions``**:  rawdata used by tower-eval for evaluations to access references.

-  **``submission_unbabel+it``**:  official submission to wmt24 chat translation task.

-  **``scripts``**: additional python and bash script to download data, generate multiple candidates, pcxmi and run mbr and contrastive decoding.


Cite our work:

```bibtex
@article{pombal2024context,
  title={A context-aware framework for translation-mediated conversations},
  author={Pombal, Jos{\'e} and Agrawal, Sweta and Fernandes, Patrick and Zaranis, Emmanouil and Martins, Andr{\'e} FT},
  journal={arXiv preprint arXiv:2412.04205},
  year={2024}
}
```

```bibtex
@inproceedings{pombal2024improving,
  title={Improving context usage for translating bilingual customer support chat with large language models},
  author={Pombal, Jos{\'e} and Agrawal, Sweta and Martins, Andr{\'e} FT},
  booktitle={Proceedings of the Ninth Conference on Machine Translation},
  pages={993--1003},
  year={2024}
}
```