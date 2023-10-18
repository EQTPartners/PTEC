import pandas as pd
import matplotlib.pyplot as plt


model_size_map = {
    "llama-7b": 7 * 10**9,
    "bloom-7b1": 7.1 * 10**9,
    "bloom-3b": 3 * 10**9,
    "bloom-1b7": 1.7 * 10**9,
    "flan-t5-small": 0.06 * 10**9,
    "flan-t5-base": 0.25 * 10**9,
    "flan-t5-large": 0.78 * 10**9,
    "flan-t5-xl": 3 * 10**9,
    "flan-t5-xxl": 11 * 10**9,
    "gpt-3.5-turbo": 175 * 10**9,
    "7b": 7 * 10**9,
}

model_name_map = {
    "llama-7b": "llama",
    "bloom-7b1": "bloom",
    "bloom-3b": "bloom",
    "bloom-1b7": "bloom",
    "flan-t5-small": "flan-t5",
    "flan-t5-base": "flan-t5",
    "flan-t5-large": "flan-t5",
    "flan-t5-xl": "flan-t5",
    "flan-t5-xxl": "flan-t5",
    "gpt-3.5-turbo": "ChatGPT",
    "7b": "vicuna",
}


color_map = {
    "llama": "tab:blue",
    "bloom": "tab:red",
    "flan-t5": "tab:green",
    "ChatGPT": "yellow",
    "vicuna": "black",
}


def results_to_df(results):
    results_organized = []
    for path, result in results.items():
        model_name = path.split("/")[3]
        row = {"model_name": model_name_map[model_name]}
        row["method"] = path.split("/")[1]
        row["augmentation"] = path.split("/")[4]
        row["model_size"] = model_size_map[model_name]
        row["macro-precision"] = result["sectors"]["macro avg"]["precision"]
        row["macro-recall"] = result["sectors"]["macro avg"]["recall"]
        row["macro-f1"] = result["sectors"]["macro avg"]["f1-score"]
        row["micro-precision"] = result["sectors"]["micro avg"]["precision"]
        row["micro-recall"] = result["sectors"]["micro avg"]["recall"]
        row["micro-f1"] = result["sectors"]["micro avg"]["f1-score"]
        row["micro-jaccard"] = result["jaccard_micro"]
        row["macro-jaccard"] = result["jaccard_macro"]
        results_organized.append(row)

    return pd.DataFrame(results_organized)


def plot_method_model_size(results, augmentation, metric):
    results = results.loc[results["augmentation"] == augmentation]
    methods = results.method.unique()
    method_map = {
        "classification_head": "-",
        "RN": "--",
        "KNN": ":",
    }
    fig, ax = plt.subplots(figsize=(6, 4))

    lines = []
    labels = []

    for model_name in results.model_name.unique():
        added_to_legend = False
        for method in methods:
            frame = results.loc[results["method"] == method]
            subframe = frame.loc[frame["model_name"] == model_name]
            (line,) = plt.plot(
                subframe["model_size"],
                subframe[metric],
                marker="o",
                linestyle=method_map[method],
                label=model_name if not added_to_legend else "_nolegend_",
                color=color_map[model_name],
            )
            if not added_to_legend:
                lines.append(plt.Line2D([0], [0], color=color_map[model_name], lw=2))
                labels.append(model_name)
                added_to_legend = True

    ax.set_title(f"{metric} by model size")
    ax.set_xlabel("Model Size")
    ax.set_ylabel(metric)

    # Create the legends
    legend1 = ax.legend(
        lines,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        title="model",
        borderaxespad=0,
    )
    legend2 = ax.legend(
        [
            plt.Line2D([0], [0], color="k", linestyle=ls, lw=2)
            for _, ls in method_map.items()
        ],
        list(method_map.keys()),
        title="Method",
        loc="upper left",
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0,
    )

    ax.add_artist(legend1)
    ax.set_xscale("log")
    plt.show()


def plot_augmentation_model_size(results, method, metric):
    results = results.loc[results["method"] == method]
    augmentation_methods = results.augmentation.unique()
    augmentation_map = {"augmented": "-", "preprocessed": "--"}

    fig, ax = plt.subplots(figsize=(6, 4))

    lines = []
    labels = []

    for model_name in results.model_name.unique():
        added_to_legend = False
        for method in augmentation_methods:
            frame = results.loc[results["augmentation"] == method]
            subframe = frame.loc[frame["model_name"] == model_name]
            (line,) = plt.plot(
                subframe["model_size"],
                subframe[metric],
                marker="o",
                linestyle=augmentation_map[method],
                label=model_name if not added_to_legend else "_nolegend_",
                color=color_map[model_name],
            )
            if not added_to_legend:
                lines.append(plt.Line2D([0], [0], color=color_map[model_name], lw=2))
                labels.append(model_name)
                added_to_legend = True

    ax.set_title(f"{metric} by model size")
    ax.set_xlabel("Model Size")
    ax.set_ylabel(metric)

    # Create the legends
    legend1 = ax.legend(
        lines,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        title="model",
        borderaxespad=0,
    )
    legend2 = ax.legend(
        [
            plt.Line2D([0], [0], color="k", linestyle=ls, lw=2)
            for _, ls in augmentation_map.items()
        ],
        list(augmentation_map.keys()),
        title="Method",
        loc="upper left",
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0,
    )

    ax.add_artist(legend1)
    ax.set_xscale("log")
    plt.show()
