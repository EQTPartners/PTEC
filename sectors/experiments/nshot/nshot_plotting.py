import pandas as pd
import matplotlib.pyplot as plt


model_size_map = {
    "decapoda-research/llama-7b-hf": 7 * 10**9,
    "bigscience/bloom-7b1": 7.1 * 10**9,
    "bigscience/bloom-3b": 3 * 10**9,
    "bigscience/bloom-1b7": 1.7 * 10**9,
    "google/flan-t5-base": 0.25 * 10**9,
    "google/flan-t5-large": 0.78 * 10**9,
    "google/flan-t5-xl": 3 * 10**9,
    "google/flan-t5-xxl": 11 * 10**9,
    "gpt-3.5-turbo": 175 * 10**9,
    "vicuna/7b": 7 * 10**9,
}

model_name_map = {
    "decapoda-research/llama-7b-hf": "llama",
    "bigscience/bloom-7b1": "bloom",
    "bigscience/bloom-3b": "bloom",
    "bigscience/bloom-1b7": "bloom",
    "google/flan-t5-base": "flan-t5",
    "google/flan-t5-large": "flan-t5",
    "google/flan-t5-xl": "flan-t5",
    "google/flan-t5-xxl": "flan-t5",
    "gpt-3.5-turbo": "ChatGPT",
    "vicuna/7b": "vicuna",
}


color_map = {
    "llama": "tab:blue",
    "bloom": "tab:red",
    "flan-t5": "tab:green",
    "ChatGPT": "yellow",
    "vicuna": "black",
}

n_map = {0: 0.1, 1: 0.3, 2: 0.5, 4: 0.7, 8: 0.9}


def results_to_df(results_list):
    results_organized = []
    for result in results_list:
        model_name = result["model_name"]
        for key in result.keys():
            if key != "model_name":
                row = {"model_name": model_name_map[model_name]}
                row["n"] = int(key[0])
                row["model_size"] = model_size_map[model_name]
                row["trie_search"] = "trie-search" in key
                row["show_sectors"] = "show-sectors" in key
                row["macro-precision"] = result[key]["sectors"]["macro avg"][
                    "precision"
                ]
                row["macro-recall"] = result[key]["sectors"]["macro avg"]["recall"]
                row["macro-f1"] = result[key]["sectors"]["macro avg"]["f1-score"]
                row["micro-precision"] = result[key]["sectors"]["micro avg"][
                    "precision"
                ]
                row["micro-recall"] = result[key]["sectors"]["micro avg"]["recall"]
                row["micro-f1"] = result[key]["sectors"]["micro avg"]["f1-score"]
                row["micro-jaccard"] = result[key]["jaccard_micro"]
                row["macro-jaccard"] = result[key]["jaccard_macro"]
                results_organized.append(row)

    return pd.DataFrame(results_organized)


def plot_n_model_size(results, metric, show_sector=True, trie_search=True):
    fig, ax = plt.subplots(figsize=(6, 4))

    lines = []
    labels = []

    for model_name in results.model_name.unique():
        for i in [0, 1, 2, 4, 8]:
            frame = results.loc[
                (results.show_sectors == show_sector)
                & (results.trie_search == trie_search)
                & (results.n == i)
            ]
            subframe = frame.loc[frame["model_name"] == model_name]
            (line,) = plt.plot(
                subframe["model_size"],
                subframe[metric],
                marker="o",
                alpha=n_map[i],
                label=(model_name, i),
                color=color_map[model_name],
            )

            if i == 0:
                lines.append(
                    plt.Line2D([0], [0], color=color_map[model_name], lw=2)
                )  # Set alpha to 1.0 (default) for lines in the legend
                labels.append(model_name)

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
        [plt.Line2D([0], [0], color="k", alpha=n_map[i], lw=2) for i in n_map],
        list(n_map.keys()),
        title="N Examples",
        loc="upper left",
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0,
    )

    # Add the first legend back to the chart
    ax.add_artist(legend1)
    ax.set_xscale("log")
    fig.tight_layout()

    # Display the chart
    plt.show()


def plot_method_model_size(results, metric, i):
    method_map = {
        "Trie Search & Show Sectors": "-",
        "Trie Search": "--",
        "Show Sectors": "-.",
        "None": ":",
    }

    method_frame_map = {
        "Trie Search & Show Sectors": results.loc[
            (results.show_sectors == True)
            & (results.trie_search == True)
            & (results.n == i)
        ],
        "Trie Search": results.loc[
            (results.show_sectors == False)
            & (results.trie_search == True)
            & (results.n == i)
        ],
        "Show Sectors": results.loc[
            (results.show_sectors == True)
            & (results.trie_search == False)
            & (results.n == i)
        ],
        "None": results.loc[
            (results.show_sectors == False)
            & (results.trie_search == False)
            & (results.n == i)
        ],
    }

    # Adjust the figure size
    fig, ax = plt.subplots(figsize=(6, 4))

    lines = []
    labels = []

    for model_name in results.model_name.unique():
        for method in method_map.keys():
            frame = method_frame_map[method]
            subframe = frame.loc[frame["model_name"] == model_name]
            (line,) = plt.plot(
                subframe["model_size"],
                subframe[metric],
                marker="o",
                linestyle=method_map[method],
                label=(model_name, i),
                color=color_map[model_name],
            )

            if method == "None":
                lines.append(plt.Line2D([0], [0], color=color_map[model_name], lw=2))
                labels.append(model_name)

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

    # Add the first legend back to the chart
    ax.add_artist(legend1)
    ax.set_xscale("log")

    # Display the chart
    plt.show()
