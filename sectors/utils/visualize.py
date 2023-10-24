import os
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import List, Dict
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sectors.config import RESULTS_DIR, FIG_DIR, DATA_DIR
from sectors.utils.save_load import open_most_recent_results


model_map = {"llama-7b": "LLaMa 7B", "bloom-1b7": "Bloom 1B7", "None": "None"}
replacement_sectors = [  # hypothetical sector names we can use for visualization to not disclose true sectors
    "Aerospace and Defense",
    "AgTech",
    "AI and Machine Learning",
    "Automobile Manufacturing",
    "Banking and Finance",
    "Biotechnology and Pharmaceuticals",
    "Blockchain and Cryptocurrency",
    "Chemicals and Materials Manufacturing",
    "Clean Energy and Renewable Resources",
    "Commercial Real Estate",
    "Construction and Infrastructure",
    "Consumer Goods and Services",
    "Cybersecurity",
    "Data Analytics and Big Data",
    "Digital Marketing and Advertising",
    "E-commerce and Online Retail",
    "EdTech",
    "EV and Autonomous Vehicles",
    "Entertainment and Media Production",
    "CleanTech",
    "Fashion and Apparel Manufacturing",
    "Food and Beverage Manufacturing",
    "FoodTech",
    "Gaming and Esports",
    "Health and Fitness Technology",
    "Healthcare Services",
    "Home and Garden Products Manufacturing",
    "Hospitality and Tourism",
    "Industrial Automation and Robotics",
    "Infrastructure and Transportation Management Software",
    "InsurTech",
    "IoT",
    "LegalTech",
    "ManuTech",
    "Medical Devices and Equipment",
    "MedTech",
    "Mining and Natural Resources",
    "Mobile Application Development",
    "Nanotechnology",
    "Non-profit and Social Impact",
    "Oil and Gas",
    "Personal Care and Cosmetics Manufacturing",
    "Pet Industry",
    "Professional Services",
    "Quantum Computing",
    "PropTech",
    "Retail and Wholesale Trade",
    "Satellite and Space Technology",
    "Semiconductor Manufacturing",
    "Shipping and Logistics",
    "Smart Home Technology",
    "Social Media and Networking Platforms",
    "SaaS",
    "Solar Energy",
    "SportsTech",
    "Supply Chain Management",
    "Telecommunications",
    "Textile and Apparel Manufacturing",
    "Travel and Accommodation Services",
    "Utilities and Energy",
    "Venture Capital and Private Equity",
    "VR/AR",
    "Waste Management and Recycling",
    "Wearable Technology",
    "Wireless Technology and Networking",
    "3D Printing and Additive Manufacturing",
    "FinTech",
    "Public Sector and Government Services",
    "Research and Development Services",
    "HRTech",
    "Nutraceuticals and Functional Foods",
    "Marine and Maritime Technology",
    "Art and Cultural Industry",
    "Publishing and Journalism",
    "Film and Television Industry",
    "Music Industry",
]
remove = [
    "id",
    "legal_name",
    "description",
    "short_description",
    "tags",
    "len_des",
    "tags_string",
    "len_tags",
    "prompt",
    "des_len",
    "ID",
    "title",
    "type",
    "message",
]


def plot_wordcloud(dataset):
    train = pd.read_json(
        os.path.join(DATA_DIR, dataset, "train_preprocessed.json"), lines=True
    )

    label_columns = [col for col in train.columns if col not in remove]
    labels = train[label_columns]

    if dataset == "industries":
        labels.columns = replacement_sectors

    wc = WordCloud(
        max_words=76,
        width=8000,
        height=4000,
        prefer_horizontal=1,
        colormap="tab20c",
        background_color="rgba(255, 255, 255, 0)",
        mode="RGBA",
        relative_scaling=1,
    )
    wc.generate_from_frequencies(labels.sum().to_dict())
    wc.to_file(os.path.join(FIG_DIR, dataset, "wordcloud.png"))

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def plot_data_insights(dataset):
    text_col = "description" if dataset == "industries" else "message"
    train = pd.read_json(os.path.join(DATA_DIR, dataset, "train.json"), lines=True)
    train["des_len"] = train.apply(lambda x: len(x[text_col]), axis=1)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.histplot(
        train.des_len,
        bins=20,
        log_scale=(False, True),
        color="#ff5703",
        edgecolor="#ff5703",
        ax=ax,
    )
    ax.set_xlabel("Description Length (#Char)", fontsize=22)
    ax.set_ylabel("Log Frequency", fontsize=22)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.75)

    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)

    plt.savefig(
        os.path.join(FIG_DIR, dataset, "des_len_og.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()

    preprocessed = pd.read_json(
    os.path.join(DATA_DIR, dataset, "train_preprocessed.json"), lines=True
    )
    preprocessed["des_len"] = preprocessed.apply(lambda x: len(x[text_col]), axis=1)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.histplot(
        preprocessed.des_len,
        bins=20,
        log_scale=(False, True),
        color="#ff5703",
        edgecolor="#ff5703",
        ax=ax,
    )
    ax.set_xlabel("Description Length (#Char)", fontsize=22)
    ax.set_ylabel("Log Frequency", fontsize=22)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.75)

    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)

    plt.savefig(
        os.path.join(FIG_DIR, dataset, "des_len_pre.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()

    label_columns = [col for col in train.columns if col not in remove]
    labels = train[label_columns]
    label_count = labels.sum(axis=1)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.histplot(
        label_count,
        bins=4,
        log_scale=(False, True),
        color="#ff5703",
        edgecolor="#ff5703",
        ax=ax,
    )
    ax.set_xlabel("#Labels per Example", fontsize=22)
    ax.set_ylabel("Log Frequency", fontsize=22)
    ax.set_xticks(np.arange(1, 5, step=1))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.75)

    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)

    plt.savefig(
        os.path.join(FIG_DIR, dataset, "label_count.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()

    # show ditribution of examples over classes
    label_columns = [col for col in train.columns if col not in remove]
    labels = train[label_columns]
    label_count = labels.sum(axis=0)
    label_count = label_count.sort_values(ascending=False)

    bar_color = "#ff5703"
    alpha_value = 0.3
    bar_color_rgb = tuple(int(bar_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    bar_color_rgba = tuple([x/255 for x in bar_color_rgb] + [alpha_value])


    fig, ax = plt.subplots(figsize=(6, 6))
    sns.barplot(
        x=label_count.index,
        y=label_count.values,
        color="#ff5703",
        alpha=0.8,
        edgecolor=bar_color_rgba,
        ax=ax,
    )
    ax.set_xlabel("#Examples per Label", fontsize=22)
    ax.set_ylabel("Frequency", fontsize=22)
    yticks = np.arange(0, label_count.max(), 100)
    ax.set_yticks(yticks)
    xticks = np.arange(0, len(label_count), 20)
    ax.set_xticks(xticks)

    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(axis='x', colors='none')
    xticks = ax.get_xticks()
    num_xticks = len(xticks)
    invisible_labels = ['1'] * num_xticks
    ax.set_xticklabels(invisible_labels, color='none')
    ax.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.75)

    plt.savefig(
        os.path.join(FIG_DIR, dataset, "label_dist.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()


def get_metrics(runs: List[Dict]):
    if len(runs) == 1:
        return runs[0]["sectors"]["macro avg"]["f1-score"], 0, 0
    else:
        runs = [run["sectors"]["macro avg"]["f1-score"] for run in runs]
        return np.mean(runs), np.std(runs), np.std(runs) / np.sqrt(len(runs))


def get_runs(results: Dict) -> List[Dict]:
    if "no_seed" in results.keys():
        return [results["no_seed"]]
    else:
        return [results["42"], results["43"], results["44"]]


def get_row(results, specs):
    runs = get_runs(results)
    mean, std, se = get_metrics(runs)

    row = pd.DataFrame(
        {
            "model": model_map[specs[-1]],
            "method": "CH" if specs[2] == "classification_head" else specs[2],
            "augmented": specs[1],
            "trie_search": specs[3],
            "training_flops": results["training_flops"]
            if "training_flops" in results.keys()
            else np.nan,
            "inference_flops": results["inference_flops"]
            if "inference_flops" in results.keys()
            else np.nan,
            "macro_f1": mean,
            "macro_f1_std": std,
            "macro_f1_se": se,
        },
        index=[0],
    )
    return row


def get_results(dataset: str):
    rows = []
    for path, _, filenames in os.walk(RESULTS_DIR / dataset):
        if len(filenames) > 0 and "results" in filenames[0] and "old" not in path:
            path = path[path.find("results/") + len("results/") :]
            specs = path.split("/")
            if specs[-1] in ["llama-7b", "bloom-1b7", "None"]:
                results = open_most_recent_results(path)
                rows.append(get_row(results, specs))

    res = pd.concat(rows, ignore_index=True)
    res.loc[res["trie_search"] == "trie_search", "method"] = (
        res.loc[res["trie_search"] == "trie_search", "method"] + " + TS"
    )
    res["method"] = pd.Categorical(
        res["method"],
        [
            "PTEC",
            "PT + TS",
            "PT",
            "CH",
            "KNN",
            "RadiusNN",
            "gzip",
            "N-shot + TS",
            "N-shot",
        ],
    )
    res.sort_values("method", inplace=True)
    return res


def plot_results(results, flop_type, error_type, loc1, loc2, dataset="industries"):
    sns.set_theme(style="whitegrid")
    results = results.loc[results["augmented"] == "preprocessed"]
    gzip = results.loc[results["method"] == "gzip"]["macro_f1"].values[0]

    # Defining the marker shapes for the methods
    marker_shapes = ["o", "v", "s", "D", "X", "d", ">", "<"]
    method_list = results["method"].unique().tolist()
    method_list.remove("gzip")
    if flop_type == "Training":
        method_list.remove("N-shot")
        method_list.remove("N-shot + TS")
    method_marker = {
        method: marker for method, marker in zip(method_list, marker_shapes)
    }

    # Defining colors for the models
    colors = ["#ff5703", "#060c31"]  # Add more colors if there are more models
    model_list = results["model"].unique().tolist()
    model_list.remove("None")
    model_colors = {model: color for model, color in zip(model_list, colors)}

    plt.figure(figsize=(8, 3))
    plt.grid(color="lightgray", linestyle="--", linewidth=0.5)

    # Creating the scatter plot
    for method in method_list:
        for model in model_list:
            sub_results = results[
                (results["method"] == method) & (results["model"] == model)
            ]
            if flop_type == "Training":
                x = sub_results["training_flops"] + 1
            elif flop_type == "Inference":
                x = sub_results["inference_flops"] + 1
            elif flop_type == "Total":
                x = sub_results["training_flops"] + sub_results["inference_flops"] + 1
            sns.scatterplot(
                x=x,
                y=sub_results["macro_f1"],
                marker=method_marker[method],
                color=model_colors[model],
                s=100,
            )

            plt.errorbar(
                x=x,
                y=sub_results["macro_f1"],
                yerr=sub_results[f"macro_f1_{error_type}"],
                fmt="none",
                ecolor="darkgrey",
                capsize=3,
            )

    plt.xscale("log")
    plt.xlabel(f"{flop_type} FLOPs (log scale)")
    plt.ylabel("Macro F1 Score")
    ax = plt.gca()
    ax.margins(x=0.02)
    xmin, xmax = ax.get_xlim()[0], ax.get_xlim()[1]
    plt.hlines(
        y=gzip,
        xmin=xmin,
        xmax=xmax,
        color=(0.5, 0.5, 0.5),
        linestyle="-",
        linewidth=1,
    )
    # (0.4, 0.4, 1)
    legend_elements_method = [
        plt.Line2D(
            [0],
            [0],
            marker=method_marker[m],
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label=m,
            markeredgewidth=0.5,
            markeredgecolor="None",
            linestyle="None",
        )
        for m in method_list
    ]
    legend_elements_model = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=model_colors[m],
            markersize=8,
            label=m,
            markeredgecolor="None",
            linestyle="None",
        )
        for m in model_list
    ]
    legend1 = plt.legend(
        handles=legend_elements_method, title="Method", loc=loc1, fontsize=8
    )
    plt.gca().add_artist(legend1)
    plt.legend(handles=legend_elements_model, title="Model", loc=loc2, fontsize=8)

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, dataset, f"performance_{flop_type}.png".lower()), dpi=300
    )
    plt.show()


def get_macro_ap(trues, preds):
    trues, preds = np.array(trues), np.array(preds)
    n_classes = trues.shape[1]
    precision = dict()
    recall = dict()
    ap = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(trues[:, i], preds[:, i])
        ap[i] = average_precision_score(trues[:, i], preds[:, i])

    # A "macro-average": quantifying score on all classes jointly
    precision["macro"], recall["macro"], _ = precision_recall_curve(
        trues.ravel(), preds.ravel()
    )
    ap["macro"] = average_precision_score(trues, preds, average="macro")
    return precision, recall, ap


def calculate_prec_rec(trues, preds):
    trues, preds = np.array(trues), np.array(preds)
    TP = np.sum((preds == 1) & (trues == 1))
    FP = np.sum((preds == 1) & (trues == 0))
    FN = np.sum((preds == 0) & (trues == 1))

    # calculate precision and recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return precision.mean(), recall.mean()


def get_macro_auc(trues, preds):
    trues, preds = np.array(trues), np.array(preds)
    n_classes = trues.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(trues[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc


def calculate_rates(trues, preds):
    trues, preds = np.array(trues), np.array(preds)
    TP = np.sum((preds == 1) & (trues == 1))
    FN = np.sum((preds == 0) & (trues == 1))
    FP = np.sum((preds == 1) & (trues == 0))
    TN = np.sum((preds == 0) & (trues == 0))

    # calculate TPR and FPR
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    return TPR.mean(), FPR.mean()


def plot_roc_pr_curves(dataset, model):
    order = ["PTEC", "CH", "KNN", "RadiusNN", "PT", "N-shot", "gzip"]
    all_results = {}
    methods_map = {}
    for path, _, filenames in os.walk(RESULTS_DIR / dataset):
        if len(filenames) > 0 and "results" in filenames[0] and "old" not in path:
            path = path[path.find("results/") + len("results/") :]
            if path.split("/")[-1] in [model, "None"]:
                results = open_most_recent_results(path)
                if (
                    "probas" in results.keys()
                    and "trues" in results.keys()
                    and "preprocessed" in path
                    and "unconstrained" in path
                ):
                    if type(results["probas"][0][0]) == int:
                        results["TPR"], results["FPR"] = calculate_rates(
                            results["trues"], results["probas"]
                        )
                        results["precision"], results["recall"] = calculate_prec_rec(
                            results["trues"], results["probas"]
                        )
                    else:
                        results["FPR"], results["TPR"], results["auc"] = get_macro_auc(
                            results["trues"], results["probas"]
                        )
                        (
                            results["precision"],
                            results["recall"],
                            results["ap"],
                        ) = get_macro_ap(results["trues"], results["probas"])
                    results["method"] = (
                        "CH"
                        if path.split("/")[2] == "classification_head"
                        else path.split("/")[2]
                    )
                    methods_map[results["method"]] = path
                    all_results[path] = results

    # Plot all ROC curves
    plt.figure()  # figsize=(5, 5))
    for path, results in all_results.items():
        if "auc" in results.keys():
            plt.plot(
                results["FPR"]["macro"],
                results["TPR"]["macro"],
                label=f"{results['method']} (AUROC = {results['auc']['macro']:0.2f})",
            )
        else:
            plt.scatter(
                results["FPR"],
                results["TPR"],
                marker="s",
                s=100,
                label=results["method"],
            )
    plt.xlabel("False Positive Rate", fontsize=20)
    plt.ylabel("True Positive Rate", fontsize=20)
    plt.legend()
    plt.ylim([0.0, 1.05])
    plt.grid(color="lightgray", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    indices = [
        labels.index(
            f"{method} (AUROC = {all_results[methods_map[method]]['auc']['macro']:0.2f})"
        )
        if "auc" in all_results[methods_map[method]].keys()
        else labels.index(method)
        for method in order
    ]
    plt.legend([handles[idx] for idx in indices], [labels[idx] for idx in indices])
    plt.savefig(os.path.join(FIG_DIR, dataset, "roc.png"))
    plt.show()

    plt.figure()  # figsize=(5, 5))
    for path, results in all_results.items():
        if "ap" in results.keys():
            plt.plot(
                results["recall"]["macro"],
                results["precision"]["macro"],
                label=f"{results['method']} (AP = {results['ap']['macro']:0.2f})",
            )
        else:
            plt.scatter(
                results["recall"],
                results["precision"],
                marker="s",
                s=100,
                label=results["method"],
            )

    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision", fontsize=20)
    # plt.legend(loc="upper right")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid(color="lightgray", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    indices = [
        labels.index(
            f"{method} (AP = {all_results[methods_map[method]]['ap']['macro']:0.2f})"
        )
        if "ap" in all_results[methods_map[method]].keys()
        else labels.index(method)
        for method in order
    ]
    plt.legend([handles[idx] for idx in indices], [labels[idx] for idx in indices])
    plt.savefig(os.path.join(FIG_DIR, dataset, "pr.png"))

    plt.show()
