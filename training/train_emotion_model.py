import torch
import json
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, roc_auc_score
from datasets import DatasetDict

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback
)

from dataset_loader import load_go_emotions
from preprocessing import preprocess_dataset


# -----------------------------
# LOAD LABEL MAPPINGS (FIXED)
# -----------------------------
def load_label_mappings():
    with open("data/labels.json", "r") as f:
        labels_data = json.load(f)

    id2label = {int(k): v for k, v in labels_data["id2label"].items()}
    label2id = labels_data["label2id"]

    return id2label, label2id


# -----------------------------
# FOCAL LOSS
# -----------------------------
def focal_loss(logits, labels, gamma=1.5):
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    probs = torch.sigmoid(logits)

    pt = labels * probs + (1 - labels) * (1 - probs)
    loss = bce * ((1 - pt) ** gamma)

    return loss.mean()


# -----------------------------
# DATA COLLATOR
# -----------------------------
class MultiLabelDataCollator(DataCollatorWithPadding):

    def __call__(self, features):
        labels = [torch.tensor(f["labels"], dtype=torch.float) for f in features]
        batch = super().__call__(features)
        batch["labels"] = torch.stack(labels)
        return batch


# -----------------------------
# CUSTOM TRAINER
# -----------------------------
class FocalTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        loss = focal_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss


# -----------------------------
# LOSS TRACKER
# -----------------------------
class LossTrackerCallback(TrainerCallback):

    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        if "loss" in logs:
            self.train_losses.append(logs["loss"])

        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])


# -----------------------------
# THRESHOLD TUNING
# -----------------------------
def tune_thresholds(model, dataset, device, num_labels):

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataset:

            input_ids = batch["input_ids"].unsqueeze(0).to(device)
            attention_mask = batch["attention_mask"].unsqueeze(0).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

            all_probs.append(probs)
            all_labels.append(batch["labels"].numpy())

    all_probs = np.array(all_probs)
    all_labels = (np.array(all_labels) > 0).astype(int)

    thresholds = []

    for i in range(num_labels):

        best_f1 = 0
        best_t = 0.5

        for t in np.arange(0.1, 0.9, 0.05):

            preds = (all_probs[:, i] >= t).astype(int)
            f1 = f1_score(all_labels[:, i], preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        thresholds.append(best_t)

    return thresholds


# -----------------------------
# ROC-AUC
# -----------------------------
def compute_roc_auc(model, dataset, device):

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataset:

            input_ids = batch["input_ids"].unsqueeze(0).to(device)
            attention_mask = batch["attention_mask"].unsqueeze(0).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

            all_probs.append(probs)
            all_labels.append(batch["labels"].numpy())

    all_probs = np.array(all_probs)
    all_labels = (np.array(all_labels) > 0).astype(int)

    return roc_auc_score(all_labels, all_probs, average="micro")


# -----------------------------
# TRAIN FUNCTION (FIXED LABELS)
# -----------------------------
def train_model(train_dataset, val_dataset):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # 🔥 LOAD LABELS PROPERLY
    id2label, label2id = load_label_mappings()
    num_labels = len(id2label)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification"
    )

    # Regularization
    model.config.dropout = 0.3
    model.config.attention_dropout = 0.3

    model.to(device)

    training_args = TrainingArguments(
        output_dir="./models",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.1,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=150,
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=150,
        fp16=torch.cuda.is_available()
    )

    loss_tracker = LossTrackerCallback()

    trainer = FocalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=MultiLabelDataCollator(tokenizer),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            loss_tracker
        ]
    )

    trainer.train()

    return model, tokenizer, loss_tracker, id2label


# -----------------------------
# LEARNING CURVE
# -----------------------------
def learning_curve_pipeline(dataset):

    TRAIN_SIZES = [200, 500, 1000, 2000, 5000]

    train_scores = []
    val_scores = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for size in TRAIN_SIZES:

        print(f"\nTraining with {size} samples...")

        subset = dataset["train"].select(range(min(size, len(dataset["train"]))))

        model, _, _, _ = train_model(subset, dataset["validation"])

        train_auc = compute_roc_auc(model, subset, device)
        val_auc = compute_roc_auc(model, dataset["validation"], device)

        train_scores.append(train_auc)
        val_scores.append(val_auc)

    plt.figure()
    plt.plot(TRAIN_SIZES, train_scores, label="Train ROC-AUC")
    plt.plot(TRAIN_SIZES, val_scores, label="Validation ROC-AUC")
    plt.xlabel("Training Size")
    plt.ylabel("ROC-AUC")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid()

    plt.savefig("learning_curve.png")
    plt.close()

    print("Saved learning curve → learning_curve.png")


# -----------------------------
# MAIN
# -----------------------------
def main():

    dataset = load_go_emotions()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    encoded_dataset = preprocess_dataset(dataset, tokenizer)

    split = encoded_dataset["train"].train_test_split(test_size=0.1, seed=42)

    dataset = DatasetDict({
        "train": split["train"],
        "validation": encoded_dataset["validation"],
        "test": split["test"]
    })

    # -----------------------------
    # TRAIN
    # -----------------------------
    model, tokenizer, loss_tracker, id2label = train_model(
        dataset["train"],
        dataset["validation"]
    )

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    SAVE_PATH = "models/emotion_classifier_v6"

    print(f"\nSaving model to {SAVE_PATH}...")

    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    print("Model saved successfully.")

    # -----------------------------
    # SAVE LABEL MAPPING
    # -----------------------------
    with open(f"{SAVE_PATH}/id2label.json", "w") as f:
        json.dump(id2label, f, indent=2)

    # -----------------------------
    # THRESHOLD TUNING
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    thresholds = tune_thresholds(
        model,
        dataset["validation"],
        device,
        len(id2label)
    )

    print("\nOptimal thresholds:\n", thresholds)

    with open(f"{SAVE_PATH}/thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    print("Thresholds saved.")

    # -----------------------------
    # LOSS CURVE
    # -----------------------------
    plt.figure()

    plt.plot(loss_tracker.train_losses, label="Train Loss")
    plt.plot(loss_tracker.eval_losses, label="Validation Loss")

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Curve")

    plt.legend()
    plt.grid()

    plt.savefig("loss_curve.png")
    plt.close()

    print("Saved loss curve → loss_curve.png")

    # -----------------------------
    # LEARNING CURVE
    # -----------------------------
    learning_curve_pipeline(dataset)


if __name__ == "__main__":
    main()