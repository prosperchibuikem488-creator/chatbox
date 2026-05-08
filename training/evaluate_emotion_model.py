import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataset_loader import load_go_emotions
from preprocessing import preprocess_dataset, load_label_schema


# -----------------------------
# Load model 
# -----------------------------
MODEL_PATH = "models/emotion_classifier_v6"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()


# -----------------------------
# Load dataset
# -----------------------------
dataset = load_go_emotions()
encoded_dataset = preprocess_dataset(dataset, tokenizer)

#  Use validation set (consistent + stable)
test_dataset = encoded_dataset["validation"]

#  IMPORTANT FIX: convert to torch tensors
test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

label_schema = load_label_schema()


# -----------------------------
# Inference
# -----------------------------
all_probs = []
all_preds = []
all_labels = []

with torch.no_grad():

    for batch in test_dataset:

        input_ids = batch["input_ids"].unsqueeze(0).to(device)
        attention_mask = batch["attention_mask"].unsqueeze(0).to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits.cpu().numpy()

        # Sigmoid for multi-label
        probs = 1 / (1 + np.exp(-logits))

        #  Tune this threshold if needed
        preds = (probs >= 0.4).astype(int)

        all_probs.append(probs[0])
        all_preds.append(preds[0])
        all_labels.append(batch["labels"].cpu().numpy())


all_probs = np.array(all_probs)
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ✅ Convert soft labels → binary
all_labels = (all_labels > 0).astype(int)


# -----------------------------
# Metrics
# -----------------------------
precision = precision_score(all_labels, all_preds, average="micro", zero_division=0)
recall = recall_score(all_labels, all_preds, average="micro", zero_division=0)

micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

hamming = hamming_loss(all_labels, all_preds)

print("\nEvaluation Results:\n")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Micro F1: {micro_f1:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Hamming Loss: {hamming:.4f}")


# -----------------------------
# Confusion Matrix (per label)
# -----------------------------
print("\nGenerating confusion matrices...\n")

for i, label in enumerate(label_schema):

    if len(np.unique(all_labels[:, i])) < 2:
        print(f"Skipping {label} (only one class present)")
        continue

    cm = confusion_matrix(
        all_labels[:, i],
        all_preds[:, i]
    )

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title(f"Confusion Matrix - {label}")
    plt.savefig(f"confusion_matrix_{label}.png")
    plt.close()


# -----------------------------
# ROC Curves
# -----------------------------
print("\nGenerating ROC curves...\n")

plt.figure()

for i, label in enumerate(label_schema):

    y_true = all_labels[:, i]
    y_score = all_probs[:, i]

    if len(np.unique(y_true)) < 2:
        print(f"Skipping {label} (only one class present)")
        continue

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{label} (AUC = {roc_auc:.2f})"
    )

# Reference line
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Emotions")

plt.legend(loc="lower right")
plt.grid()

plt.savefig("roc_curve_all_emotions.png")
plt.close()

print("Saved ROC curve -> roc_curve_all_emotions.png")