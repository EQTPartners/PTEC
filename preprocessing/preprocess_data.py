import torch
import pandas as pd
from transformers import pipeline
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from sectors.utils.models import create_model, set_seed
from sectors.config import DATA_DIR

pd.options.mode.chained_assignment = None


def preprocess_industry_data(
    dataset: pd.DataFrame, max_char_len: int = 1000
) -> pd.DataFrame:
    dataset.description = dataset.apply(
        lambda row: row["short_description"].replace("\n", " ")
        if len(row["description"]) > max_char_len
        and type(row["short_description"]) == str
        else row["description"].replace("\n", " "),
        axis=1,
    )
    dataset["tags_string"] = dataset.tags.apply(
        lambda x: ", ".join(x).lower().replace("\n", " ")
    )
    return dataset


def summarize_column(
    column: pd.Series,
    batch_size: int,
    max_token_len: int,
    summarizer: pipeline,
    generation_config: dict,
    max_char_len: int = 1000,
) -> pd.Series:
    leng = column.apply(lambda x: len(x))
    long_des = column.loc[leng > 1000].tolist()
    print(f"Summarizing {len(long_des)} of {len(column)} texts.")
    print("Examples of original texts:")
    for des in long_des[:5]:
        print(des)
        print("----------------------------")
    data = DataLoader(long_des, batch_size=batch_size)
    sum_des = []
    for batch in data:
        summary = summarizer(
            batch, max_length=max_token_len, min_length = 50, generation_config=generation_config
        )
        sum_des += [out["summary_text"] for out in summary]
    column.loc[leng > max_char_len] = sum_des
    print("Examples of generated summaries:")
    for des in sum_des[:5]:
        print(des)
        print("----------------------------")
    return column


def summarize_keywords(
    column: pd.Series,
    max_char_len: int,
    max_token_len: int,
    tokenizer: object,
    model: object,
    generation_config: dict,
) -> pd.Series:
    leng = column.apply(lambda x: len(x))
    long_tags = column.loc[leng > max_char_len].tags_string.tolist()
    data = DataLoader(long_tags, batch_size=2)
    sum_tags = []
    for batch in data:
        text = [
            b
            + "\nFrom the options given above, select a set of 25 most informative and distinct keywords"
            for b in batch
        ]
        inputs = tokenizer.batch_encode_plus(
            text,
            return_tensors="pt",
            padding=True,
            max_length=2048,
            truncation=True,
        )
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_token_len,
            generation_config=generation_config,
            do_sample=True,
            top_k=50,
        )
        sum_tags += tokenizer.batch_decode(outputs, skip_special_tokens=True)
    column.loc[leng > max_char_len] = sum_tags
    return column


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
    model_name = "google/flan-t5-large"
    max_char_len = 1000
    batch_size = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, generation_config = create_model(model_name, False, device, False)

    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        truncation=True,
        device=0,
        penalty_alpha=0.8,
        top_k=4,
        no_repeat_ngram_size=3,
    )

    for dataset_name in ["train", "dev", "test"]:
        path = f"{DATASET_DIR}/{dataset_name}.json"
        dataset = pd.read_json(path, lines=True)

        if args.dataset == "industries":
            max_token_len = 150
            dataset = preprocess_industry_data(dataset, max_char_len)
            dataset.description = summarize_column(
                dataset.description,
                batch_size,
                max_token_len,
                summarizer,
                generation_config,
                max_char_len
            )
            dataset.tags_string = summarize_keywords(dataset.tags_string)

            dataset["prompt"] = (
                "Company name: "
                + dataset.legal_name
                + ". Keywords: "
                + dataset.tags_string
                + ". Description: "
                + dataset.description
                + " This company classifies into the sector(s):"
            )
        elif args.dataset == "hatespeech":
            max_token_len = 200
            dataset.message = summarize_column(
                dataset.message,
                batch_size,
                max_token_len,
                summarizer,
                generation_config,
                max_char_len
            )
            dataset["prompt"] = (
                "Title: "
                + dataset.title
                + " Message: "
                + dataset.message
                + " This message classifies into the following types of hate speech:"
            )

        save_path = f"{DATASET_DIR}/{dataset_name}_preprocessed.json"
        dataset.to_json(save_path, orient="records", lines=True, index=True)


if __name__ == "__main__":
    set_seed(42)
    main()
