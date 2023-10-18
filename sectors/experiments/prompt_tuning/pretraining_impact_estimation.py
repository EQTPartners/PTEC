import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
from torch.utils.data import DataLoader
from sklearn.metrics import multilabel_confusion_matrix
from sectors.utils.models import set_seed
from sectors.config import DATA_DIR
from sectors.experiments.prompt_tuning.model import (
    get_trainer_class,
    get_tuner_class,
    construct_generation_args,
)


def get_correct_prediction_percentage(arr):
    correct_predictions = np.diagonal(arr, axis1=-2, axis2=-1).sum(axis=-1)
    total_predictions = arr.sum(axis=(-2, -1))
    percentages = correct_predictions / total_predictions
    return percentages


if __name__ == "__main__":
    args = construct_generation_args()
    results = json.load(open(os.path.join(args.path, "results.json")))
    orig_args = results["args"]
    args.seed = orig_args["seed"]
    args.dataset = orig_args["dataset"]
    args.model_name = orig_args["model_name"]
    args.batch_size = orig_args["batch_size"]
    args.load_in_8bit = orig_args["load_in_8bit"]
    args.num_beams = orig_args["num_beams"]
    args.sp_len = orig_args["sp_len"]
    args.labels = orig_args["labels"]
    args.head = orig_args["head"]
    args.optimizer = orig_args["optimizer"]
    args.TRAINER_CLASS = get_trainer_class(args.head)
    args.TUNE_CLASS = get_tuner_class(args.head)
    set_seed(args.seed)
    trainer = args.TRAINER_CLASS(args)

    # load checkpoint and data
    checkpoint = {
        "softprompt": torch.load(os.path.join(args.path, "softprompt.pth")),
        "classification_head": torch.load(
            os.path.join(args.path, "classification_head.pth")
        ),
    }
    trainer.load_checkpoint(checkpoint)
    TEST_PATH = join(DATA_DIR, args.dataset, "test_pretraining_knowledge.json")
    trainer.test_set = trainer.get_dataset(TEST_PATH, args)
    trainer.test_loader = DataLoader(
        trainer.test_set, batch_size=trainer.args.batch_size
    )

    # extract model performance
    trainer.model.eval()
    with torch.no_grad():
        trues = []
        preds = []
        for inputs, labels in tqdm(trainer.test_loader):
            trues_array, predictions_array, _ = trainer.evaluate_pass(inputs, labels)
            trues += trues_array
            preds += predictions_array
    c = multilabel_confusion_matrix(trues, preds, samplewise=True)
    pred_correct = get_correct_prediction_percentage(c)

    # save performance to data
    test = pd.read_json(TEST_PATH, lines=True)
    test["pred_correct"] = pred_correct
    test.to_json(TEST_PATH, orient="records", lines=True, index=True)

    report = {}
    # evaluate companies with pretraining knowledge
    knowledge = trainer.test_data[trainer.test_data["pretraining_knowledge"]]
    trainer.test_set = trainer.get_dataset(knowledge, trainer.args)
    trainer.test_loader = DataLoader(
        trainer.test_set, batch_size=trainer.args.batch_size
    )
    report["knowledge"] = trainer.evaluate_score(
        0, trainer.test_loader, "Test", trie_search=True
    )

    # evaluate companies without pretraining knowledge
    no_knowledge = trainer.test_data[~trainer.test_data["pretraining_knowledge"]]
    test_set = trainer.get_dataset(no_knowledge, trainer.args)
    test_loader = DataLoader(test_set, batch_size=trainer.args.batch_size)
    report["no_knowledge"] = trainer.evaluate_score(
        0, test_loader, "Test", trie_search=True
    )

    # evaluate complete test set
    test_set = trainer.get_dataset(trainer.test_data, trainer.args)
    test_loader = DataLoader(test_set, batch_size=trainer.args.batch_size)
    report["all"] = trainer.evaluate_score(0, test_loader, "Test", trie_search=True)

    json.dump(
        report, open(join(DATA_DIR, args.dataset, "pretraining_impact.json"), "w")
    )
