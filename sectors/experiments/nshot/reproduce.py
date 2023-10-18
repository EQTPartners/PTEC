import torch
import argparse
from torch.utils.data import Subset

from sectors.utils.models import create_model
from sectors.utils.dataset import MultiLabelT2TDataset
from sectors.experiments.nshot.nshot_model import nshot
from sectors.utils.save_load import save_results_with_timestamp, get_path
from sectors.config import DATA_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="hatespeech",
        choices=["hatespeech", "industries"],
    )
    parser.add_argument("--model_name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--trie_search", type=bool, default=False)
    parser.add_argument("--show_sectors", type=bool, default=False)
    parser.add_argument("--labels", type=str, default="sectors")
    parser.add_argument("--load_in_8bit", action="store_true")
    args = parser.parse_args()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = MultiLabelT2TDataset("train_preprocessed.json", args)
    test = MultiLabelT2TDataset("test_preprocessed.json", args)
    model_trio = create_model(args.model_name, False, device, False)

    results = {"args": vars(args)}
    for i in range(42, 45):
        results[i], _, _ = nshot(
            args.model_name,
            model_trio=model_trio,
            n=args.n,
            trie_search=args.trie_search,
            show_sectors=args.show_sectors,
            train=train,
            test=test,
            labels=train.labels.columns,
            max_tokens=50,
            device=device,
        )
    # calculate flops
    test = Subset(test, range(100))
    with torch.profiler.profile(with_flops=True) as inference_profiler:
        nshot(
            args.model_name,
            model_trio=model_trio,
            n=args.n,
            trie_search=args.trie_search,
            show_sectors=args.show_sectors,
            train=train,
            test=test,
            labels=test.labels,
            max_tokens=50,
            device=device,
        )
    # inference FLOPs for 10M (100 * 1e4)
    results["inference_flops"] = (
        sum(event.flops for event in inference_profiler.key_averages()) * 1e4
    )
    results["training_flops"] = 0

    PATH = get_path(
        args.dataset,
        "preprocessed",
        "N-shot",
        "trie_search" if args.trie_search else "unconstrained",
        args.model_name,
    )
    save_results_with_timestamp(results, PATH)
