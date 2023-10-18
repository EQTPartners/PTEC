import os
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from os.path import join
from typing import List, Tuple
from torch.utils.data import DataLoader

from sectors.utils.models import create_model
from sectors.config import DATA_DIR, RESULTS_DIR


def get_sequence_embeddings(
    batch: List[str], tokenizer, model, device: str, normalize: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = tokenizer.batch_encode_plus(
        batch, return_tensors="pt", padding=True, max_length=2048, truncation=True
    )
    attention_mask = inputs["attention_mask"]
    if hasattr(model, "encoder"):
        outputs = model.encoder(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=attention_mask.to(device),
        )
    else:
        outputs = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=attention_mask.to(device),
        )
    last_hidden_state = outputs.last_hidden_state
    last_hidden_state = last_hidden_state.detach().to("cpu")

    if normalize:  # Bloom embeddings are normalized to avoid overflow
        min_values = torch.min(last_hidden_state, dim=1, keepdim=True).values
        max_values = torch.max(last_hidden_state, dim=1, keepdim=True).values
        last_hidden_state = (last_hidden_state - min_values) / (max_values - min_values)

    # Masking padding tokens
    masked_hidden_state = last_hidden_state * attention_mask.unsqueeze(-1)

    # sum and mean pooling
    sum_pooling = torch.sum(masked_hidden_state, 1)
    mean_pooling = sum_pooling / torch.sum(attention_mask, dim=1, keepdim=True)

    # Last token (last non-padding token)
    sequence_lengths = (
        attention_mask.sum(dim=1) - 1
    )  # minus 1 because we're doing zero-based indexing
    last_token = last_hidden_state[
        torch.arange(last_hidden_state.size(0)), sequence_lengths
    ]

    return mean_pooling, sum_pooling, last_token


def process_batches(
    batches: torch.utils.data.DataLoader, tokenizer, model, device, normalize
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_mean_pooling = []
    all_sum_pooling = []
    all_last_token = []

    for batch in tqdm(batches):
        mean_pooling, sum_pooling, last_token = get_sequence_embeddings(
            batch, tokenizer, model, device, normalize
        )

        all_mean_pooling.append(mean_pooling)
        all_sum_pooling.append(sum_pooling)
        all_last_token.append(last_token)

    all_mean_pooling = torch.cat(all_mean_pooling, dim=0)
    all_sum_pooling = torch.cat(all_sum_pooling, dim=0)
    all_last_token = torch.cat(all_last_token, dim=0)

    return all_mean_pooling, all_sum_pooling, all_last_token


def generate_embeddings(args, model, tokenizer, device, normalize):
    for dataset_name in ["train", "dev", "test"]:
        if dataset_name == "train":
            path = f"{DATA_DIR}/{args.dataset}/{dataset_name}_{args.augmented}.json"
        else:
            path = f"{DATA_DIR}/{args.dataset}/{dataset_name}_preprocessed.json"
        dataset = pd.read_json(path, lines=True)
        data = DataLoader(dataset["prompt"], batch_size=args.batch_size, shuffle=False)
        model.eval()
        all_mean_pooling, all_sum_pooling, all_last_token = process_batches(
            data, tokenizer, model, device, normalize
        )
        directory = f"{DATA_DIR}/{args.dataset}/embeddings/{args.model_name}"
        os.makedirs(directory, exist_ok=True)
        if dataset_name == "train":
            torch.save(
                all_mean_pooling,
                f"{directory}/{dataset_name}_{args.augmented}_mean_pooling.pt",
            )
            torch.save(
                all_sum_pooling,
                f"{directory}/{dataset_name}_{args.augmented}_sum_pooling.pt",
            )
            torch.save(
                all_last_token,
                f"{directory}/{dataset_name}_{args.augmented}_last_token.pt",
            )
        else:
            torch.save(
                all_mean_pooling,
                f"{directory}/{dataset_name}_preprocessed_mean_pooling.pt",
            )
            torch.save(
                all_sum_pooling,
                f"{directory}/{dataset_name}_preprocessed_sum_pooling.pt",
            )
            torch.save(
                all_last_token, f"{directory}/{dataset_name}_preprocessed_last_token.pt"
            )


def estimate_FLOPs(args, model, tokenizer, device, normalize):
    TRAIN_PATH = join(DATA_DIR, args.dataset, "train_preprocessed.json")
    dataset = pd.read_json(TRAIN_PATH, lines=True).sample(n=args.n)
    data = DataLoader(
        dataset["prompt"].tolist(), batch_size=args.batch_size, shuffle=False
    )
    model.eval()
    with torch.profiler.profile(with_flops=True) as embedding_profiler:
        process_batches(data, tokenizer, model, device, normalize)
    embedding_flops = (
        sum(event.flops for event in embedding_profiler.key_averages()) / args.n
    )

    # save flops
    flops = {"embedding_flops": embedding_flops}
    path = join(
        RESULTS_DIR,
        args.dataset,
        "embedding_flops",
        args.model_name,
        "embedding_flops.json",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(flops, f, indent=4)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="hatespeech",
        choices=["hatespeech", "industries"],
    )
    parser.add_argument("--model_name", type=str, default="huggyllama/llama-7b")
    parser.add_argument(
        "--augmented", type=str, default="preprocessed", choices=["preprocessed", "augmented"]
    )
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    model, tokenizer, _ = create_model(
        args.model_name, True, device, load_in_8bit=args.load_in_8bit
    )
    normalize = True if "bloom" in args.model_name else False

    generate_embeddings(args, model, tokenizer, device, normalize)
    estimate_FLOPs(args, model, tokenizer, device, normalize)
