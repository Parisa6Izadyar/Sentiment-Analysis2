"""Classification metrics and saved confusion matrices."""
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score


def score_model(model, features, labels):
    predicted = model.predict(features)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
    else:
        scores = model.predict_proba(features)[:, 1]
    report = {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(labels, predicted, labels=[0, 1],
                                                        target_names=["negative", "positive"],
                                                        output_dict=True, zero_division=0),
    }
    return report, predicted


def plot_confusions(metrics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
    for ax, (name, values) in zip(axes.ravel(), metrics.items()):
        matrix = np.asarray(values["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for (row, col), count in np.ndenumerate(matrix):
            ax.text(col, row, str(count), ha="center", va="center",
                    color="white" if count > matrix.max()/2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["negative", "positive"],
               yticklabels=["negative", "positive"], xlabel="Predicted", ylabel="Actual",
               title=f"{name.replace('_', ' ')}\naccuracy {values['accuracy']:.3f}; macro F1 {values['macro_f1']:.3f}")
    fig.savefig(Path(path), dpi=150)
    plt.close(fig)
