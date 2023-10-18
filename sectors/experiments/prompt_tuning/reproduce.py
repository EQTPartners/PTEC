import json
import torch
from os.path import join
from sectors.utils.models import set_seed
from sectors.experiments.prompt_tuning.model import (
    get_trainer_class,
    get_tuner_class,
    construct_generation_args,
)
from sectors.utils.save_load import open_most_recent_model


if __name__ == "__main__":
    args = construct_generation_args()
    results = json.load(open(join(args.path, "results.json")))
    orig_args = results["args"]
    args.seed = orig_args["seed"]
    args.dataset = orig_args["dataset"]
    args.model_name = orig_args["model_name"]
    args.augmented = orig_args["augmented"]
    args.batch_size = orig_args["batch_size"]
    args.load_in_8bit = orig_args["load_in_8bit"]
    args.num_beams = orig_args["num_beams"]
    args.sp_len = orig_args["sp_len"]
    args.labels = orig_args["labels"]
    args.head = orig_args["head"]
    args.optimizer = orig_args["optimizer"]
    args.TRAINER_CLASS = get_trainer_class(args.head)
    args.TUNE_CLASS = get_tuner_class(args.head)
    true_epochs = args.epochs
    args.epochs = 1
    set_seed(args.seed)
    trainer = args.TRAINER_CLASS(args)

    # load checkpoint and evaluate test set
    checkpoint = {"softprompt": open_most_recent_model(args.path, "softprompt_")}
    if args.head == "ch":
        checkpoint["classification_head"] = {
            "classification_head": open_most_recent_model(
                args.path, "classification_head_"
            )
        }
        report_reproduced = trainer.evaluate_test(checkpoint, False)
        results["reproduced_test_results"] = report_reproduced
        trues, probs = trainer.collect_probabilities(trainer.test_loader)
        results["test_probs"] = probs
        results["test_trues"] = trues
    else:
        report_trie_search = trainer.evaluate_test(checkpoint, True)
        report_unconstrained_decoding = trainer.evaluate_test(checkpoint, False)
        results["reproduced_test_results_trie_search"] = report_trie_search
        results[
            "reproduced_test_results_unconstrained_decoding"
        ] = report_unconstrained_decoding
        trues, probs = trainer.collect_predictions(trainer.test_loader, False)
        results["test_probs"] = probs
        results["test_trues"] = trues
        trues, probs = trainer.collect_predictions(trainer.test_loader, True)
        results["test_probs_trie_search"] = probs
        results["test_trues_trie_search"] = trues

    # profile flops
    if args.profile_flops:
        # profile training
        factor = 100
        with torch.profiler.profile(with_flops=True) as training_profiler:
            trainer.test_run(factor)
        results["training_flops"] = (
            sum(event.flops for event in training_profiler.key_averages())
            * factor
            * true_epochs
        )

        # profile inference with trie search
        factor = 10
        with torch.profiler.profile(with_flops=True) as inference_ts_profiler:
            trainer.test_evaluation(0, trainer.test_loader, "Test", True, factor)
        results["inference_flops_trie_search"] = (
            sum(event.flops for event in inference_ts_profiler.key_averages()) * factor
        )

        # profile inference without trie search
        with torch.profiler.profile(with_flops=True) as inference_profiler:
            trainer.test_evaluation(0, trainer.test_loader, "Test", False, factor)
        results["inference_flops"] = (
            sum(event.flops for event in inference_profiler.key_averages()) * factor
        )

    path = join(args.path, "results.json")
    json.dump(results, open(path, "w"), indent=4)
