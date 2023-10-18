import torch
import random
import numpy as np
from typing import List
from fastchat.model import load_model
from transformers import (
    AutoModel,
    AutoTokenizer,
    GPT2LMHeadModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    LlamaConfig,
    BloomModel,
    BloomTokenizerFast,
    BloomConfig,
    GenerationConfig,
    AutoModelForSequenceClassification,
)
from parallelformers import parallelize


sigmoid = torch.nn.Sigmoid()


def create_model(
    model_name: str,
    encoder_only: bool,
    device: str,
    load_in_8bit: bool,
    num_labels: int = None,
    parallelize: bool = False
):
    if "vicuna" in model_name:
        num_gpus = 1
        max_gpu_memory = 15000000 #40000000
        cpu_offloading = False
        debug = False
        model, tokenizer = load_model(
            model_name,
            device,
            num_gpus,
            max_gpu_memory,
            load_in_8bit,
            cpu_offloading,
            debug=debug,
        )
        generation_config = GenerationConfig.from_pretrained(model_name)
        tokenizer.col_token_id = 1125
        tokenizer.sep_token_id = 2056
        tokenizer.eos_token_id = generation_config.eos_token_id
        tokenizer.bos_token_id = generation_config.bos_token_id
        tokenizer.pad_token_id = 0
    else:
        MODEL_CLASS, TOKENIZER_CLASS = get_model_and_tokenizer_class(
            model_name, encoder_only, num_labels
        )
        if load_in_8bit:
            if num_labels:
                model = MODEL_CLASS.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    problem_type="multi_label_classification",
                    device_map="auto",
                    load_in_8bit=True,
                    torch_dtype=torch.float16,
                )
            else:
                model = MODEL_CLASS.from_pretrained(
                    model_name,
                    device_map="auto",
                    load_in_8bit=True,
                    torch_dtype=torch.float16,
                )
        else:
            if num_labels:
                model = MODEL_CLASS.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    problem_type="multi_label_classification",
                    torch_dtype=torch.float16
                ).to(device)
            else:
                model = MODEL_CLASS.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        if parallelize:
            parallelize(model, num_gpus=2, fp16=True, verbose='detail')
        if (
            "cuda" in device
            and not load_in_8bit
            and not num_labels
            # and "bloom" not in model_name
        ):
            model = model.half()
        tokenizer = TOKENIZER_CLASS.from_pretrained(model_name, padding_side="right")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.col_token_id = None
        if "llama" in model_name or "vicuna" in model_name:
            config = LlamaConfig()
            generation_config = GenerationConfig.from_model_config(config)
            tokenizer.col_token_id = 29901 # 1125
            tokenizer.sep_token_id = 2056
            tokenizer.eos_token_id = generation_config.eos_token_id
            tokenizer.bos_token_id = generation_config.bos_token_id
            tokenizer.pad_token_id = 0
        elif "bloom" in model_name:
            config = BloomConfig()
            generation_config = GenerationConfig.from_model_config(config)
            generation_config.pad_token_id = generation_config.eos_token_id
            tokenizer.sep_token_id = 30
            tokenizer.col_token_id = 29
        else:
            generation_config = GenerationConfig.from_pretrained(model_name)
        if "flan-t5" in model_name:
            tokenizer.bos_token_id = 0
            tokenizer.sep_token_id = 117
        if "gpt" in model_name:
            tokenizer.sep_token_id = 26
            tokenizer.col_token_id = 25
            tokenizer.pad_token_id = 0
    return model, tokenizer, generation_config


def get_model_and_tokenizer_class(model_name: str, encoder_only: bool, num_labels: int):
    if num_labels:
        return AutoModelForSequenceClassification, AutoTokenizer
    if encoder_only:
        if "bloom" in model_name:
            return BloomModel, BloomTokenizerFast
        return AutoModel, AutoTokenizer
    if "llama" in model_name or "vicuna" in model_name:
        return LlamaForCausalLM, LlamaTokenizer
    elif "gpt" in model_name:
        return GPT2LMHeadModel, AutoTokenizer
    elif "flan-t5" in model_name:
        return AutoModelForSeq2SeqLM, AutoTokenizer
    elif "bloom" in model_name:
        return AutoModelForCausalLM, AutoTokenizer
    else:
        raise NotImplementedError(
            "This model type ``{}'' is not implemented.".format(model_name)
        )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def ordered_set(input_list: List[str]) -> List[str]:
    seen = dict()
    output_list = []
    for item in input_list:
        if item not in seen:
            seen[item] = True
            output_list.append(item)
    return output_list


def get_predictions(out: torch.Tensor) -> List[List[float]]:
    return [(sigmoid(o) > 0.5).float().tolist() for o in out]
