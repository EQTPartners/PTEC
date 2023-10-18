import gzip
import numpy as np
import pandas as pd
from os.path import join
from argparse import ArgumentParser
from sectors.config import DATA_DIR
from sectors.utils.evaluation import sector_report
from sectors.utils.dataset import get_labels
from sectors.utils.save_load import get_path, save_results_with_timestamp


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="hatespeech",
        choices=["hatespeech", "industries"],
    )
    args = parser.parse_args()

    TEST_PATH = join(DATA_DIR, args.dataset, "test_preprocessed.json")
    test = pd.read_json(TEST_PATH, lines=True)
    TRAIN_PATH = join(DATA_DIR, args.dataset, "train_preprocessed.json")
    train = pd.read_json(TRAIN_PATH, lines=True)
    labels = get_labels(train).columns

    trues = []
    preds = []
    for row in test.iterrows():
        x1 = row[1]["prompt"]
        Cx1 = len(gzip.compress(x1.encode()))
        distance_from_x1 = []
        for x2 in train["prompt"]:
            Cx2 = len(gzip.compress(x2.encode()))
            x1x2 = " ".join([x1, x2])
            Cx1x2 = len(gzip.compress(x1x2.encode()))
            ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)

            distance_from_x1.append(ncd)

        sorted_idx = np.argsort(np.array(distance_from_x1))
        pred = train.iloc[sorted_idx[0]][labels].to_numpy()
        true = row[1][labels].to_numpy().astype(int)
        trues.append(true.tolist())
        preds.append(pred.tolist())

    report = sector_report(trues, preds, labels)
    results = {
        "no_seed": report,
        "trues": trues,
        "probas": preds,
    }

    PATH = get_path(args.dataset, "preprocessed", "gzip", "unconstrained", "None/None")
    save_results_with_timestamp(results, PATH)


if __name__ == "__main__":
    main()
