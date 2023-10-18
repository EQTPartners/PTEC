import os
import json
import pandas as pd
from dotenv import load_dotenv
from typing import List, Tuple
from argparse import ArgumentParser
from sklearn.model_selection import GroupShuffleSplit
from sectors.utils.models import set_seed
from sectors.utils.dataset import stratified_split
from sectors.config import DATA_DIR

load_dotenv()


def grouped_split(
    df: pd.DataFrame, xcol: List[str], ycol: List[str], test_size=0.4, seed=42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train and test sets, ensuring that each title only appears in one split.
    """
    df = df.reset_index(drop=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)

    groups, _ = pd.factorize(df["title"])
    train_idx, test_idx = next(gss.split(df[xcol], df[ycol], groups=groups))
    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]
    return train_df, test_df


def fill_and_drop(orgs: pd.DataFrame) -> pd.DataFrame:
    orgs.loc[orgs.description.isna(), "description"] = orgs.loc[
        orgs.description.isna()
    ].short_description
    orgs.loc[orgs.legal_name.isna(), "legal_name"] = orgs.loc[
        orgs.legal_name.isna()
    ].name
    orgs = orgs.dropna(subset=["description", "legal_name"])
    return orgs


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="hatespeech",
        choices=["hatespeech", "industries"],
    )
    args = parser.parse_args()

    DATASET_DIR = DATA_DIR / args.dataset
    TRAIN_PATH = DATASET_DIR / "train.json"
    TEST_PATH = DATASET_DIR / "test.json"
    DEV_PATH = DATASET_DIR / "dev.json"
    KEYWORD_MAP_PATH = DATASET_DIR / "label_map_keyword.json"
    LABEL_MAP_PATH = DATASET_DIR / "label_map_label.json"

    if args.dataset == "industries":
        ORGS_PATH = DATASET_DIR / "organizations_labelled.json"
        SECTORS_PATH = DATASET_DIR / "sector_specs.json"
        # removing high level sectors since these are not very informative
        remove = os.getenv("TOP_LEVEL_SECTORS").split(",")

        orgs = pd.read_json(ORGS_PATH, lines=True)
        sector_specs = pd.read_json(SECTORS_PATH)

        print(f"Total number of instances before filtering: {len(orgs)}")
        print(f"Total number of sectors before filtering: {len(sector_specs)}")

        # compose prompt
        orgs = fill_and_drop(orgs)
        orgs = orgs.merge(
            sector_specs, how="left", left_on="sector_id", right_on="sector_id"
        )
        orgs = orgs.dropna(subset=["sector_name"])
        orgs = orgs[~orgs.sector_name.isin(remove)]

        # only keep the sectors which occur at least 20 times
        sector_counts = orgs["sector_id"].value_counts()
        sectors_to_keep = sector_counts[sector_counts >= 20].index.tolist()
        orgs = orgs[orgs["sector_id"].isin(sectors_to_keep)]

        one_hot_labels = pd.get_dummies(orgs.sector_name)
        dataset = pd.concat(
            [
                orgs.id,
                orgs.legal_name,
                orgs.description,
                orgs.short_description,
                orgs.tags,
                one_hot_labels,
            ],
            axis=1,
        )
        dataset = (
            dataset.groupby(["id", "legal_name", "description"]).max().reset_index()
        )
        xcol = ["id", "legal_name", "description", "short_description", "tags"]
        ycol = orgs.sector_name.unique()

        keyword_map = dict(zip(sector_specs.sector_name, sector_specs.keywords))
        label_map = dict(zip(sector_specs.sector_name, sector_specs.sector_name))

        train, val_test = stratified_split(dataset, xcol, ycol, 0.25)
        val, test = stratified_split(val_test, xcol, ycol, 0.6)

    elif args.dataset == "hatespeech":
        DATA_PATH = DATASET_DIR / "hatespeech.csv"
        hatespeech = pd.read_csv(DATA_PATH)
        hatespeech = hatespeech.dropna(subset=["message"])

        # reorganize columns
        subs = ["Class", "Sub 1", "Sub 2", "Sub 3", "Sub 4", "Sub 5"]
        unique_labels = set()
        for index, row in hatespeech.iterrows():
            for sub in subs:
                label = row[sub]
                if pd.notnull(label) and label != "Hateful":
                    # If the label is not in unique_labels, add a new column for it and initialize with 0
                    if label not in unique_labels:
                        hatespeech[label] = 0
                        unique_labels.add(label)
                    # Update the corresponding column with 1 if the label is present
                    hatespeech.at[index, label] = 1

        print(f"Total number of instances before filtering: {len(hatespeech)}")
        print(f"Total number of sectors before filtering: {len(unique_labels)}")

        # Find columns where the sum of values is less than 20
        cols_to_drop = [
            col
            for col in hatespeech.columns
            if hatespeech[col].dtype == "int64" and hatespeech[col].sum() < 20
        ]
        hatespeech.drop(columns=subs + cols_to_drop, inplace=True)
        labels = [
            col
            for col in hatespeech.columns
            if hatespeech[col].dtype == "int64" and col != "ID"
        ]
        hatespeech = hatespeech[hatespeech[labels].sum(axis=1) != 0]

        dataset = hatespeech

        xcol = ["ID", "title", "message"]
        ycol = [col for col in dataset.columns if dataset[col].dtype == "int64"]

        label_map = keyword_map = dict(zip(ycol, ycol))

        for seed in range(1000):
            train, val_test = grouped_split(dataset, xcol, ycol, 0.25, seed)
            val, test = grouped_split(val_test, xcol, ycol, 0.6, seed)
            if (
                val[ycol].sum().sort_values()[0] > 1
                and test[ycol].sum().sort_values()[0] > 2
            ):
                break
        
        for df in [train, val, test]:
            df["title"].fillna("None", inplace=True)

    print(f"Unique labels: {len(ycol)}")
    # at least 3 examples per sector in the test set, at least 2 in tha val set
    print(f"Instances in training set: {len(train)}")
    print(f"Instances in validation set: {len(val)}")
    print(f"Instances in test set: {len(test)}")
    assert val[ycol].sum().sort_values()[0] > 1
    assert test[ycol].sum().sort_values()[0] > 2

    # save
    train.to_json(TRAIN_PATH, orient="records", lines=True, index=True)
    val.to_json(DEV_PATH, orient="records", lines=True, index=True)
    test.to_json(TEST_PATH, orient="records", lines=True, index=True)
    with open(KEYWORD_MAP_PATH, "w") as fp:
        json.dump(keyword_map, fp)
    with open(LABEL_MAP_PATH, "w") as fp:
        json.dump(label_map, fp)


if __name__ == "__main__":
    set_seed(42)
    main()
