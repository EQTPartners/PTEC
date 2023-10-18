import torch
import random
import pandas as pd
from tqdm import tqdm
from typing import List
from argparse import ArgumentParser
from fastchat.model import load_model, get_conversation_template
from sectors.utils.dataset import get_labels
from sectors.utils.models import set_seed
from sectors.config import DATA_DIR


device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "helloollel/vicuna-7b"
num_gpus = 1
max_gpu_memory = 40000000 # 40000000
load_8bit = True
cpu_offloading = False
debug = False
max_new_tokens = 300


def get_augmentations(
    example: str, ratio: int, label_string: str, model: object, tokenizer: object, dataset: str
) -> List[str]:
    if dataset == "industries":
        ending = " This company classifies into the sector(s):"
        task = f"Make up another company which has the same business model and operates in the sector(s) {label_string}. Follow the same structure as in the example: "
    else:
        ending = " This message classifies into the following types of hate speech:"
        task = f"Make up another social media comment which has the same meaning and classifies into these categories of hate speech: {label_string}. Follow the same structure as in the example: "
    example = example[: -len(ending)]
    msg = task + example

    conv = get_conversation_template(model_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=0.8,
        num_return_sequences=ratio,
        max_new_tokens=max_new_tokens,
    )

    augmentations = []
    for output in output_ids:
        output = output[len(input_ids[0]) :]
        output = tokenizer.decode(
            output, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        if dataset == "industries":
            if (
                "Company name: " in output
                and "Description: " in output
                and "Keywords: " in output
            ):
                output = output.replace("\n", " ")
                start_index = output.find("Company name: ")
                output = output[start_index:]
                augmentations.append(output + ending)
        else:
            if (
                "Title: " in output
                and "Message: " in output
            ):
                output = output.replace("\n", " ")
                start_index = output.find("Title: ")
                output = output[start_index:]
                augmentations.append(output + ending)
    return augmentations


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="hatespeech",
        choices=["hatespeech", "industries"],
    )
    args = parser.parse_args()
    model, tokenizer = load_model(
        model_path,
        device,
        num_gpus,
        max_gpu_memory,
        load_8bit,
        cpu_offloading,
        debug=debug,
    )

    TRAIN_PATH = DATA_DIR / args.dataset / "train_preprocessed.json"
    TRAIN_AUGMENTED_PATH = DATA_DIR / args.dataset/ "train_augmented.json"

    train = pd.read_json(TRAIN_PATH, lines=True)
    labels = get_labels(train)

    train_augmented = train.copy()
    max_n = min(labels.sum().max(), 300)
    for sector in labels.columns:
        count = train[sector].sum()
        if count < max_n:
            ratio = max_n // count
            to_append = max_n - count
            appended = 0
            minority_class = train.loc[train[sector] == True]
            pbar = tqdm(total=to_append)
            while appended < to_append:
                example_index = random.sample(range(0, len(minority_class)), 1)
                labels_dict = minority_class.iloc[example_index[0]][labels.columns].to_dict()
                example_prompt = minority_class.iloc[example_index[0]].prompt
                sector_string = ", ".join(
                    minority_class[labels.columns]
                    .columns[minority_class[labels.columns].iloc[example_index[0]].astype(bool)]
                    .tolist()
                )
                augmentations = get_augmentations(
                    example_prompt, ratio, sector_string, model, tokenizer, args.dataset
                )
                for augmented_prompt in augmentations:
                    train_augmented = pd.concat(
                        [
                            train_augmented,
                            pd.Series(
                                {
                                    "prompt": augmented_prompt,
                                    **labels_dict,
                                    "augmented": True,
                                }
                            )
                            .to_frame()
                            .T,
                        ]
                    )
                    appended += 1
                    pbar.update(1)
            pbar.close()

    train_augmented = train_augmented.sample(frac=1)
    train_augmented.to_json(
        TRAIN_AUGMENTED_PATH, orient="records", lines=True, index=True
    )


if __name__ == "__main__":
    set_seed(42)
    main()
