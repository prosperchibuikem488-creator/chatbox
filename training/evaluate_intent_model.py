import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.preprocessing import label_binarize


# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "models/intent_classifier_v4"
DATASET_PATH = "intent_dataset_v5.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load dataset
# -----------------------------
def load_dataset(path=DATASET_PATH):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]

    label_list = sorted(list(set(labels)))

    label2id = {
        label: i for i, label in enumerate(label_list)
    }

    id2label = {
        i: label for label, i in label2id.items()
    }

    encoded_labels = [
        label2id[label] for label in labels
    ]

    dataset = Dataset.from_dict({
        "text": texts,
        "label": encoded_labels
    })

    return dataset, label_list, label2id, id2label


# -----------------------------
# Predict
# -----------------------------
def predict(model, tokenizer, dataset, device):

    all_preds = []
    all_probs = []
    all_labels = []

    model.eval()

    with torch.no_grad():

        for sample in dataset:

            inputs = tokenizer(
                sample["text"],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }

            outputs = model(**inputs)

            logits = outputs.logits

            probs = torch.softmax(logits, dim=1)

            pred = torch.argmax(probs, dim=1)

            all_probs.append(
                probs.cpu().numpy()[0]
            )

            all_preds.append(
                pred.cpu().numpy()[0]
            )

            all_labels.append(
                sample["label"]
            )

    return (
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_labels)
    )


# -----------------------------
# Overall Metrics
# -----------------------------
def compute_metrics(labels, preds):

    accuracy = accuracy_score(labels, preds)

    precision_macro = precision_score(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    precision_weighted = precision_score(
        labels,
        preds,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        labels,
        preds,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        labels,
        preds,
        average="weighted",
        zero_division=0
    )

    print("\n==============================")
    print("Overall Evaluation Metrics")
    print("==============================\n")

    print(f"Accuracy: {accuracy:.4f}")

    print("\n--- Macro Metrics ---")
    print(f"Precision: {precision_macro:.4f}")
    print(f"Recall:    {recall_macro:.4f}")
    print(f"F1 Score:  {macro_f1:.4f}")

    print("\n--- Weighted Metrics ---")
    print(f"Precision: {precision_weighted:.4f}")
    print(f"Recall:    {recall_weighted:.4f}")
    print(f"F1 Score:  {weighted_f1:.4f}")


# -----------------------------
# Confusion Matrix PER LABEL
# -----------------------------
def generate_per_label_confusion_matrices(
    labels,
    preds,
    label_names
):

    print("\nGenerating per-label confusion matrices...\n")

    binarized_labels = label_binarize(
        labels,
        classes=list(range(len(label_names)))
    )

    binarized_preds = label_binarize(
        preds,
        classes=list(range(len(label_names)))
    )

    for i, label in enumerate(label_names):

        y_true = binarized_labels[:, i]
        y_pred = binarized_preds[:, i]

        if len(np.unique(y_true)) < 2:
            print(f"Skipping {label} (only one class present)")
            continue

        cm = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )

        disp.plot()

        plt.title(f"Confusion Matrix - {label}")

        plt.savefig(
            f"intent_confusion_matrix_{label}.png"
        )

        plt.close()

    print("Saved all per-label confusion matrices.")


# -----------------------------
# ROC Curves
# -----------------------------
def plot_roc_curves(
    all_labels,
    all_probs,
    label_names
):

    print("\nGenerating ROC curves...\n")

    binarized = label_binarize(
        all_labels,
        classes=list(range(len(label_names)))
    )

    plt.figure()

    for i in range(len(label_names)):

        y_true = binarized[:, i]
        y_score = all_probs[:, i]

        if len(np.unique(y_true)) < 2:
            print(f"Skipping {label_names[i]}")
            continue

        fpr, tpr, _ = roc_curve(
            y_true,
            y_score
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{label_names[i]} (AUC = {roc_auc:.2f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curves for All Intents")

    plt.legend(loc="lower right")

    plt.grid()

    plt.savefig("intent_roc_curves.png")

    plt.close()

    print("Saved ROC curve -> intent_roc_curves.png")


# -----------------------------
# Main
# -----------------------------
def main():

    print("Using device:", device)

    # Load dataset
    dataset, label_names, label2id, id2label = load_dataset()

    # Load tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.to(device)

    # Predict
    preds, probs, labels = predict(
        model,
        tokenizer,
        dataset,
        device
    )

    # Overall metrics
    compute_metrics(labels, preds)

    # Classification report
    from sklearn.metrics import classification_report

    print("\n==============================")
    print("Classification Report")
    print("==============================\n")

    print(
        classification_report(
            labels,
            preds,
            target_names=label_names
        )
    )

    # Per-label confusion matrices
    generate_per_label_confusion_matrices(
        labels,
        preds,
        label_names
    )

    # ROC curves
    plot_roc_curves(
        labels,
        probs,
        label_names
    )


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    main()