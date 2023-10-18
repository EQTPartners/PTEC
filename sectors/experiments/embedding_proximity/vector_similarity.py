import torch
import argparse
import numpy as np
from typing import Dict
from functools import partial
from bayes_opt import BayesianOptimization
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier

from sectors.utils.evaluation import sector_report
from sectors.utils.dataset import load_embedding_data
from sectors.utils.save_load import (
    save_results_with_timestamp,
    get_path,
    open_embedding_flops,
)


def get_model(type: str, param: float):
    if type == "RadiusNN":
        return RadiusNeighborsClassifier(
            radius=param, weights="distance", outlier_label="most_frequent"
        )
    elif type == "KNN":
        n_neighbors = int(param)
        return KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")


def get_preds(
    param: float,
    args: argparse.Namespace,
    train_embeds: torch.Tensor,
    train_labels: torch.LongTensor,
    val_embeds: torch.Tensor,
    val_labels: torch.LongTensor,
    columns: list,
) -> Dict:
    model = get_model(args.type, param)
    model.fit(train_embeds, train_labels)
    preds = model.predict(val_embeds)
    return sector_report(val_labels, preds, columns)["sectors"]["macro avg"]["f1-score"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="hatespeech",
        choices=["hatespeech", "industries"],
    )
    parser.add_argument("--model_name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--type", type=str, default="KNN", choices=["RadiusNN", "KNN"])
    parser.add_argument(
        "--augmented",
        type=str,
        default="preprocessed",
        choices=["preprocessed", "augmented"],
    )
    parser.add_argument("--embedding_type", type=str, default="mean_pooling")
    args = parser.parse_args()

    train_embeds, train_labels, _ = load_embedding_data(
        args.dataset, "train", args.model_name, args.augmented, args.embedding_type
    )
    val_embeds, val_labels, _ = load_embedding_data(
        args.dataset, "dev", args.model_name, args.augmented, args.embedding_type
    )
    test_embeds, test_labels, columns = load_embedding_data(
        args.dataset, "test", args.model_name, args.augmented, args.embedding_type
    )

    bounds = (1, 100)
    params = {"param": bounds}
    get_preds_partial = partial(
        get_preds,
        args=args,
        train_embeds=train_embeds,
        train_labels=train_labels,
        val_embeds=val_embeds,
        val_labels=val_labels,
        columns=columns,
    )
    bayesopt = BayesianOptimization(
        get_preds_partial,
        params,
        random_state=42,
        allow_duplicate_points=True,
        verbose=0,
    )
    bayesopt.maximize(init_points=25, n_iter=15)
    param = bayesopt.max["params"]["param"]
    print(param)

    if args.type == "RadiusNN":
        if param < 5:
            bounds = (0.1, 5)
            params = {"param": bounds}
            bayesopt.set_bounds(new_bounds=params)
            bayesopt.maximize(init_points=25, n_iter=15)
            param = bayesopt.max["params"]["param"]
            print(param)
        elif param > 95:
            bounds = (95, 150)
            params = {"param": bounds}
            bayesopt.set_bounds(new_bounds=params)
            bayesopt.maximize(init_points=25, n_iter=15)
            param = bayesopt.max["params"]["param"]
            print(param)

    model = get_model(args.type, param)
    model.fit(train_embeds, train_labels)
    preds = model.predict(test_embeds)
    probas = model.predict_proba(test_embeds)
    report = sector_report(test_labels, preds, columns)
    report["trues"] = np.array(test_labels).tolist()
    report["probas"] = np.array(probas)[
        :, :, 1
    ].T.tolist()  # [proba.tolist() for proba in probas]
    args.param = param

    embedding_flops = open_embedding_flops(args)
    results = {
        "no_seed": report,
        "trues": report["trues"],
        "probas": report["probas"],
        "args": vars(args),
        # FLOPs for embedding the training set of 4149 companies
        "training_flops": embedding_flops * train_embeds.size(0),
        # FLOPs for embedding the inference set of 10M companies
        # plus FLOPs needed to calculate distances
        # TODO /20 needed??
        "inference_flops": embedding_flops * 1e6
        + 3 * train_embeds.size(1) * 1e6 * train_embeds.size(0),
    }
    PATH = get_path(
        args.dataset, args.augmented, args.type, "unconstrained", args.model_name
    )
    save_results_with_timestamp(results, PATH)


if __name__ == "__main__":
    main()
