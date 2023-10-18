import numpy as np
from typing import List, Dict, Union
from sklearn.metrics import classification_report, jaccard_score


def sector_report(
    ytrue: Union[np.ndarray, List],
    ypred: Union[np.ndarray, List],
    labels: List[str],
    display: bool = False,
) -> Dict:
    if type(ytrue) != np.ndarray:
        ytrue = np.array(ytrue).astype(bool)
    if type(ypred) != np.ndarray:
        ypred = np.array(ypred).astype(bool)

    report = {}
    # get classification report by sector
    report_string = classification_report(
        y_true=ytrue, y_pred=ypred, target_names=labels, zero_division=0
    )
    report["sectors"] = classification_report(
        y_true=ytrue,
        y_pred=ypred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    # get jaccard scores
    report["jaccard_micro"] = jaccard_score(
        y_true=ytrue, y_pred=ypred, average="micro", zero_division=0
    )
    report["jaccard_macro"] = jaccard_score(
        y_true=ytrue, y_pred=ypred, average="macro", zero_division=0
    )
    report["jaccard_weighted"] = jaccard_score(
        y_true=ytrue, y_pred=ypred, average="weighted", zero_division=0
    )
    report["jaccard_samples"] = jaccard_score(
        y_true=ytrue, y_pred=ypred, average="samples", zero_division=0
    )

    report["avg_n_predicted"] = np.mean(np.sum(ypred, axis=1))
    report["avg_n_labels"] = np.mean(np.sum(ytrue, axis=1))

    report_string += f"\nMicro Jaccard: {report['jaccard_micro']}"
    report_string += f"\nMacro Jaccard: {report['jaccard_macro']}"
    report_string += f"\nWeighted Jaccard: {report['jaccard_weighted']}"
    report_string += f"\nSamples Jaccard: {report['jaccard_samples']}"

    # investigate sectors with bad performance
    analysis = "\nSectors with precision or recall < 0.5:"
    for index, sector in enumerate(report["sectors"]):
        if "avg" not in sector:
            metrics = report["sectors"][sector]
            if metrics["precision"] < 0.5 or metrics["recall"] < 0.5:
                analysis += f"\n\n{sector}:"
                analysis += f" Precision: {metrics['precision']}"
                analysis += f" Recall: {metrics['recall']}"

                # false negatives
                fn_true = ytrue[(ytrue[:, index] == 1) & (ypred[:, index] == 0)]
                fn_pred = ypred[(ytrue[:, index] == 1) & (ypred[:, index] == 0)]
                if len(fn_true) > 0:
                    false_negatives = np.logical_and(fn_true == 0, fn_pred == 1)
                    false_negatives_count = np.sum(false_negatives, axis=0)
                    sorted_indices = np.argsort(false_negatives_count)
                    top_3_indices = sorted_indices[-3:]

                    analysis += f"\nTop 3 sectors predicted instead of {sector}:"
                    for i in reversed(top_3_indices):
                        analysis += f"\n{labels[i]}: {false_negatives_count[i]}"

                # false positives
                fp_true = ytrue[(ytrue[:, index] == 0) & (ypred[:, index] == 1)]
                fp_pred = ypred[(ytrue[:, index] == 0) & (ypred[:, index] == 1)]
                if len(fp_true) > 0:
                    false_positives = np.logical_and(fp_true == 1, fp_pred == 0)
                    false_positives_count = np.sum(false_positives, axis=0)
                    sorted_indices = np.argsort(false_positives_count)
                    top_3_indices = sorted_indices[-3:]

                    analysis += f"\nTop 3 sectors which should have been predicted instead of {sector}:"
                    for i in reversed(top_3_indices):
                        analysis += f"\n{labels[i]}: {false_positives_count[i]}"

    report["analysis"] = analysis
    if display:
        print(report_string)
    return report
