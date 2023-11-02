import os
import json
import torch
import numpy as np
from datetime import datetime
from sectors.config import MODELS_DIR, RESULTS_DIR


def get_path(
    dataset: str, augmentation: str, method: str, trie_search: str, model: str
):
    """
    Get the path to the results directory for a given experiment.

    Args:
    - dataset (str): The name of the dataset.
    - augmentation (str): augmenteed or original (preprocessed).
    - method (str): The name of the embedding method.
    - trie_search (str): Whether or not trie search was used.
    - model (str): The name of the model.

    Returns:
    - str: The composed directory path.
    """
    return os.path.join(
        dataset,
        augmentation,
        method,
        trie_search,
        model,
    )


def save_model_with_timestamp(state_dict, path: str, prefix: str = ""):
    """
    Save a PyTorch model to a file with a timestamp in its filename.

    Args:
    - model: The PyTorch model to save.
    - directory (str): The directory where the file should be saved.
    - prefix (str): The prefix of the model filename.
    """
    directory = os.path.join(MODELS_DIR, path)
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}{timestamp}.pth"
    filepath = os.path.join(directory, filename)
    torch.save(state_dict, filepath)


def open_most_recent_model(path: str, prefix: str = ""):
    """
    Open the most recent PyTorch model from a given directory.

    Args:
    - directory (str): The directory to search for the model.
    - prefix (str): The prefix of the model filename.

    Returns:
    - model: The PyTorch model from the most recent file.
    """
    directory = os.path.join(MODELS_DIR, path)
    files = os.listdir(directory)
    if prefix:
        files = [f for f in files if f.startswith(prefix) and f.endswith(".pth")]
    else:
        files = [f for f in files if f.endswith(".pth")]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)

    if files:
        return torch.load(os.path.join(directory, files[0]))
    else:
        raise ValueError(f"No PyTorch models found in directory: {directory}")


def save_results_with_timestamp(data: dict, path: str):
    """
    Save a JSON object to a file with a timestamp in its filename.

    Args:
    - data (dict): The JSON object to save.
    - directory (str): The directory where the file should be saved.
    """
    directory = os.path.join(RESULTS_DIR, path)
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.json"
    filepath = os.path.join(directory, filename)
    with open(filepath, "w") as file:
        json.dump(json_compatible(data), file, indent=4)
    print("Saved results to ", filepath)


def open_most_recent_results(path: str):
    """
    Open the most recent JSON file from a given directory.

    Args:
    - directory (str): The directory to search for the file.

    Returns:
    - dict: The JSON object from the most recent file.
    """
    directory = os.path.join(RESULTS_DIR, path)
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)

    if files:
        with open(os.path.join(RESULTS_DIR, directory, files[0]), "r") as file:
            return json.load(file)
    else:
        raise ValueError(f"No JSON files found in directory: {directory}")


def open_embedding_flops(args):
    with open(
        os.path.join(
            RESULTS_DIR,
            args.dataset,
            "embedding_flops",
            args.model_name,
            "embedding_flops.json",
        ),
        "r",
    ) as f:
        # FLOPs needed for embedding 1 company information
        return json.load(f)["embedding_flops"]


def json_compatible(data):
    if isinstance(data, list):
        data = [json_compatible(item) for item in data]
        return [item for item in data if is_json_serializable(item)]
    elif isinstance(data, dict):
        data = {key: json_compatible(value) for key, value in data.items()}
        return {
            key: value for key, value in data.items() if is_json_serializable(value)
        }
    elif isinstance(data, np.int64):
        return int(data)
    elif isinstance(data, np.float32):
        return float(data)
    else:
        return data


def is_json_serializable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError) as e:
        print(f"Error message: {e}")
        return False
