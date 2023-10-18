import json
import torch
import pandas as pd
from os.path import join
from typing import List, Tuple
from torch.utils.data import Dataset
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sectors.config import DATA_DIR


def stratified_split(
    df: pd.DataFrame, xcol: List[str], ycol: List[str], test_size=0.4
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train and test sets, stratified by the labels.
    """
    df = df.reset_index(drop=True)

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=42
    )

    train_idx, test_idx = next(msss.split(df[xcol], df[ycol]))
    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]
    return train_df, test_df


class PromptClassificationHeadDataset(Dataset):
    def __init__(self, file: str, args, n: int = None):
        super().__init__()
        self.args = args
        self.n = n
        self.__get_data_inputs_labels(file, args)
        self.targets = self.labels.to_numpy()
        self.weights = self.labels.sum().max() / self.labels.sum()

    def __get_data_inputs_labels(self, file, args):
        path = join(DATA_DIR, args.dataset, file)
        if self.n:
            self.data = pd.read_json(path, lines=True).sample(self.n)
        else:
            self.data = pd.read_json(path, lines=True)
        self.inputs = self.data["prompt"].tolist()
        self.labels = get_labels(self.data)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i], self.targets[i]


class MultiLabelT2TDataset(PromptClassificationHeadDataset):
    def __init__(self, file: str, args, n: int = None):
        super().__init__(file, args, n)
        with open(DATA_DIR / f"{args.dataset}/label_map_{args.labels}.json", "r") as f:
            self.label_map = json.load(f)
        self.labels.rename(columns=self.label_map, inplace=True)
        self.labels_sorted = self.labels.sum().sort_values(ascending=False)
        self.data["label_string"] = self.labels.apply(
            lambda row: self.__generate_label_string(row), axis=1
        )
        self.targets = self.data["label_string"].tolist()

    def __generate_label_string(self, row):
        labels = row[row == True].index.tolist()
        labels_sorted = sorted(labels, key=lambda x: -self.labels_sorted[x])
        return "; ".join(labels_sorted)


def load_embedding_data(
    dataset: str, split: str, model_name: str, augmented: str, embedding_type: str
) -> Tuple[torch.Tensor, torch.LongTensor, list]:
    if split != "train":
        augmented = "preprocessed"
    embeds = torch.load(
        join(
            DATA_DIR,
            dataset,
            "embeddings",
            model_name,
            f"{split}_{augmented}_{embedding_type}.pt",
        )
    ).to(dtype=torch.float32)
    data = pd.read_json(
        join(DATA_DIR, dataset, f"{split}_{augmented}.json"), lines=True
    )
    labels = get_labels(data)
    label_columns = list(labels.columns)
    labels = torch.tensor(labels.to_numpy()).to(torch.float32)  # .long()
    return embeds, labels, label_columns


def get_labels(dataframe: pd.DataFrame, *args) -> pd.DataFrame:
    possible_remove = [
        "id",
        "legal_name",
        "description",
        "short_description",
        "tags",
        "len_des",
        "tags_string",
        "len_tags",
        "prompt",
        "augmented",
        "pred_correct",
        "pretraining_knowledge",
        "ID",
        "type",
        "title",
        "message",
    ]
    remove = [col for col in possible_remove if col in dataframe.columns]
    return dataframe.drop(remove, axis=1)
