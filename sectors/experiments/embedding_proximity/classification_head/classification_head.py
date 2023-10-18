import argparse
import numpy as np
from bayes_opt import BayesianOptimization
from sectors.experiments.embedding_proximity import train, train_evaluate_final
from sectors.utils.save_load import (
    get_path,
    save_results_with_timestamp,
    open_embedding_flops,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="hatespeech", choices=["hatespeech", "industries"]
)
parser.add_argument("--model_name", type=str, default="huggyllama/llama-7b")
parser.add_argument("--augmented", type=str, default="preprocessed", choices=["preprocessed", "augmented"])
parser.add_argument("--embedding_type", type=str, default="mean_pooling")
parser.add_argument("--batch_size", type=int, default=128)
args = parser.parse_args()


def train_mean(lr: float, weight_decay: float) -> float:
    res = []
    for i in range(3):
        res.append(
            train(
                args.dataset,
                lr,
                weight_decay,
                args.batch_size,
                200,
                "mean_pooling",
                args.model_name,
                args.augmented,
            )
        )
    return np.mean(np.array(res))


def log_function(log_lr: float, log_weight_decay: float) -> float:
    lr = 10**log_lr
    weight_decay = 10**log_weight_decay
    return train_mean(lr, weight_decay)


if __name__ == "__main__":
    params = {
        "log_lr": (np.log10(0.000001), np.log10(0.001)),
        "log_weight_decay": (np.log10(0.000000001), np.log10(0.001)),
    }
    bayesopt = BayesianOptimization(
        log_function, params, random_state=42, allow_duplicate_points=True, verbose=0
    )
    bayesopt.maximize(init_points=25, n_iter=15)
    params_10_power = {k: 10**v for k, v in bayesopt.max["params"].items()}
    print(params_10_power)

    if (
        bayesopt.max["params"]["log_lr"] > -3.5
        and bayesopt.max["params"]["log_weight_decay"] < -7
    ):
        params = {"log_lr": (np.log10(0.0001), np.log10(1))}

        def log_function(log_lr):
            lr = 10**log_lr
            return train_mean(lr, 0)

        bayesopt = BayesianOptimization(
            log_function,
            params,
            random_state=42,
            allow_duplicate_points=True,
            verbose=0,
        )
        bayesopt.maximize(init_points=25, n_iter=15)
        params_10_power = {k: 10**v for k, v in bayesopt.max["params"].items()}
        print(params_10_power)

    elif bayesopt.max["params"]["log_lr"] > -3.5:
        params = {
            "log_lr": (np.log10(0.0001), np.log10(1)),
            "log_weight_decay": (np.log10(0.000000001), np.log10(0.001)),
        }
        bayesopt.set_bounds(new_bounds=params)
        bayesopt.maximize(init_points=25, n_iter=15)
        params_10_power = {k: 10**v for k, v in bayesopt.max["params"].items()}
        print(params_10_power)

    # training final model
    args.lr = params_10_power["log_lr"]
    args.weight_decay = (
        params_10_power["log_weight_decay"]
        if "log_weight_decay" in params_10_power
        else 0
    )

    # run 3 times
    all_results = {}
    for i in range(42, 45):
        flops, probas = (True, True) if i == 42 else (False, False)
        all_results[i], n_train, n_test = train_evaluate_final(args, flops, probas)
    embedding_flops = open_embedding_flops(args)
    all_results["args"] = vars(args)
    # training FLOPs + FLOPs needed to calculate embeddings for the training set
    all_results["training_flops"] = all_results[42]["training_flops"] + (
        embedding_flops * n_train
    )
    # inference FLOPs needed on test set projected onto 10M
    # + FLOPs needed to calculate embeddings for 10M
    all_results["inference_flops"] = (
        all_results[42]["inference_flops"] * (1e6 / n_test)
    ) + (embedding_flops * 1e6)
    all_results["probas"] = all_results[42]["probas"]
    all_results["trues"] = all_results[42]["trues"]

    PATH = get_path(
        args.dataset,
        args.augmented,
        "CH",
        "unconstrained",
        args.model_name,
    )
    save_results_with_timestamp(all_results, PATH)
