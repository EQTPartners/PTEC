import gc
import json
import torch
import numpy as np
from copy import deepcopy
from functools import partial
from bayes_opt import BayesianOptimization
from sectors.utils.models import set_seed
from sectors.experiments.prompt_tuning.model import construct_generation_args
from sectors.utils.save_load import get_path, save_results_with_timestamp


def train_and_eval(args, lr, epochs, sp_len, **kwargs):
    args.lr = 10**lr
    args.epochs = round(epochs)
    args.sp_len = round(sp_len)
    if "weight_decay" in kwargs and args.head == "ch":
        args.weight_decay = 10 ** kwargs["weight_decay"]
    if "ch_lr" in kwargs and args.differential_lr and args.head == "ch":
        args.ch_lr = 10 ** kwargs["ch_lr"]
    trainer = args.TRAINER_CLASS(args)
    result = trainer.train()
    return result["sectors"]["macro avg"]["f1-score"]


if __name__ == "__main__":
    args = construct_generation_args()
    set_seed(args.seed)
    params = {
        "lr": (np.log10(1e-9), np.log10(args.max_lr)),
        "sp_len": (50, args.max_sp_len),
        "epochs": (args.min_epochs, args.max_epochs),
    }
    if args.head == "ch":
        params["weight_decay"] = (np.log10(1e-9), np.log10(0.5))
    if args.differential_lr and args.head == "ch":
        params["ch_lr"] = (np.log10(1e-9), np.log10(0.1))

    # Bayesian optimization
    train_and_eval_partial = partial(train_and_eval, args=args)
    bayesopt = BayesianOptimization(
        train_and_eval_partial, params, random_state=42, allow_duplicate_points=True
    )
    if args.continue_from:
        prev_results = json.load(open(args.continue_from, "r"))
        for obs in prev_results["searched_params"]:
            bayesopt.register(
                params=obs["params"],
                target=obs["target"],
            )
    bayesopt.maximize(init_points=args.init_points, n_iter=args.n_iter)

    # transform best params
    params = bayesopt.max["params"]
    args.lr = 10 ** params["lr"]
    args.sp_len = round(params["sp_len"])
    args.epochs = round(params["epochs"])
    if args.head == "ch":
        args.weight_decay = 10 ** params["weight_decay"]
    if args.differential_lr and args.head == "ch":
        args.ch_lr = 10 ** params["ch_lr"]

    # Test best model with 3 different seeds
    unconstrained_results = {"searched_params": bayesopt.res, "args": vars(args)}
    if args.head == "lh":
        ts_results = unconstrained_results.copy()
    for seed in [42, 43, 44]:
        iter_args = deepcopy(args)
        iter_args.seed = seed
        trainer = iter_args.TRAINER_CLASS(iter_args)
        trainer.train()
        trainer.best_ckpt["test_report"] = trainer.evaluate_test(
            trainer.best_ckpt, False
        )
        unconstrained_results[seed] = trainer.best_ckpt["test_report"]
        if trainer.args.head == "lh":
            trainer.best_ckpt["test_report_trie_search"] = trainer.evaluate_test(
                trainer.best_ckpt, True
            )
            ts_results[seed] = trainer.best_ckpt["test_report_trie_search"]

        # profile flops
        if seed == 42:
            # training flops
            factor = 50 if trainer.args.augmented == "preprocessed" else 200
            with torch.profiler.profile(with_flops=True) as training_profiler:
                trainer.test_run(factor)
            trainer.best_ckpt["training_flops"] = (
                sum(event.flops for event in training_profiler.key_averages())
                * factor
                * trainer.args.epochs
            )
            unconstrained_results["training_flops"] = trainer.best_ckpt[
                "training_flops"
            ]

            factor = 20
            # inference flops without trie search
            with torch.profiler.profile(with_flops=True) as inf_profiler:
                trainer.test_evaluation(0, trainer.test_loader, "Test", False, factor)
            ifs = sum(event.flops for event in inf_profiler.key_averages())
            trainer.best_ckpt["inference_flops"] = (
                (ifs * factor) / len(trainer.test_loader)
            ) * 1e6
            unconstrained_results["inference_flops"] = trainer.best_ckpt[
                "inference_flops"
            ]

            # collect probs
            if args.head == "ch":
                trues, probas = trainer.collect_probabilities(trainer.test_loader)
                unconstrained_results["probas"] = probas
                unconstrained_results["trues"] = trues
            elif args.head == "lh":
                trues, probas = trainer.collect_predictions(trainer.test_loader, False)
                unconstrained_results["probas"] = probas
                unconstrained_results["trues"] = trues

                # inference flops with trie search
                with torch.profiler.profile(with_flops=True) as inf_ts_profiler:
                    trainer.test_evaluation(
                        0, trainer.test_loader, "Test", True, factor
                    )
                ifts = sum(event.flops for event in inf_ts_profiler.key_averages())
                trainer.best_ckpt["inference_flops_trie_search"] = (
                    (ifts * factor) / len(trainer.test_loader)
                ) * 1e6
                ts_results["inference_flops"] = trainer.best_ckpt[
                    "inference_flops_trie_search"
                ]
                ts_results["training_flops"] = trainer.best_ckpt["training_flops"]

                # collect probas
                trues, probas = trainer.collect_predictions(trainer.test_loader, True)
                ts_results["probas"] = probas
                ts_results["trues"] = trues

        trainer.save(trainer.best_ckpt)

        # Free memory
        del trainer
        gc.collect()

    # Save results
    UNCONSTRAINED_PATH = get_path(
        args.dataset,
        args.augmented,
        "PTEC" if args.head == "ch" else "PT",
        "unconstrained",
        args.model_name,
    )
    save_results_with_timestamp(unconstrained_results, UNCONSTRAINED_PATH)
    if args.head == "lh":
        TS_PATH = get_path(
            args.dataset,
            args.augmented,
            "PTEC" if args.head == "ch" else "PT",
            "trie_search",
            args.model_name,
        )
        save_results_with_timestamp(ts_results, TS_PATH)
