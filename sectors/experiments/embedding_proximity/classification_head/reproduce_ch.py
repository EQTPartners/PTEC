import os
import argparse
from sectors.experiments.embedding_proximity import train_evaluate_final
from sectors.utils.save_load import get_path, save_results_with_timestamp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="hatespeech",
        choices=["hatespeech", "industries"],
    )
    parser.add_argument("--model_name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--augmented", type=str, default="preprocessed", choices=["preprocessed", "augmented"])
    parser.add_argument("--embedding_type", type=str, default="mean_pooling")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0031)
    parser.add_argument("--weight_decay", type=float, default=0)
    args = parser.parse_args()

    results, _, _ = train_evaluate_final(args)

    PATH = get_path(
        args.dataset,
        args.augmented,
        "classification_head",
        "unconstrained",
        args.model_name,
    )
    save_results_with_timestamp(
        results, os.path.join(PATH, "reproduction")
    )
