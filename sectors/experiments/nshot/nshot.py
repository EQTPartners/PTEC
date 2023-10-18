import torch
import argparse
from functools import partial
from torch.utils.data import Subset
from bayes_opt import BayesianOptimization

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
    parser.add_argument("--trie_search", type=bool, default=False)
    parser.add_argument("--labels", type=str, default="label")
    parser.add_argument("--init_points", type=int, default=8)
    parser.add_argument("--n_iter", type=int, default=4)
    parser.add_argument("--load_in_8bit", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = MultiLabelT2TDataset("train_preprocessed.json", args)
    test = MultiLabelT2TDataset("test_preprocessed.json", args)
    model_trio = create_model(args.model_name, False, device, False)

    params = {"n": (-0.5, 8.5), "show_labels": (0, 1)}
    nshot_partial = partial(
        nshot,
        model_name=args.model_name,
        model_trio=model_trio,
        trie_search=args.trie_search,
        train=train,
        test=test,
        labels=train.labels.columns,
        max_tokens=50,
        device=device,
    )

    def nshot_bayes(n, show_labels):
        n = round(n)
        show_labels = bool(round(show_labels))
        return nshot_partial(n=n, show_labels=show_labels)[0]["sectors"]["macro avg"]["f1-score"]

    bayesopt = BayesianOptimization(
        nshot_bayes, params, random_state=42, allow_duplicate_points=True
    )
    bayesopt.maximize(init_points=args.init_points, n_iter=args.n_iter)
    params = bayesopt.max["params"]
    args.n = round(params["n"])
    args.show_labels = bool(round(params["show_labels"]))

    # evaluate final parameters with 3 runs
    results = {"searched_params": bayesopt.res, "args": vars(args)}
    for i in range(42, 45):
        results[i], trues, probas = nshot(
            args.model_name,
            model_trio=model_trio,
            n=args.n,
            trie_search=args.trie_search,
            show_labels=args.show_labels,
            train=train,
            test=test,
            labels=train.labels.columns,
            max_tokens=50,
            device=device
        )
        if i == 42:
            results["trues"] = trues
            results["probas"] = probas

    # calculate flops
    sample_test = MultiLabelT2TDataset("test_preprocessed.json", args, 10)
    with torch.profiler.profile(with_flops=True) as inference_profiler:
        nshot(
            args.model_name,
            model_trio=model_trio,
            n=args.n,
            trie_search=args.trie_search,
            show_labels=args.show_labels,
            train=train,
            test=sample_test,
            labels=train.labels.columns,
            max_tokens=50,
            device=device,
        )
    # inference FLOPs for 10M (10 * 1e5)
    results["inference_flops"] = (
        sum(event.flops for event in inference_profiler.key_averages()) * 1e5
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
