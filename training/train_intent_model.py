import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
    EarlyStoppingCallback
)

# -----------------------------
# LOAD DATA
# -----------------------------
def load_dataset(path="intent_dataset_v5.json"):

    with open(path, "r", encoding = "utf-8") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]

    return texts, labels


# -----------------------------
# METRICS
# -----------------------------
def compute_metrics(eval_pred):

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0)
    }


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
# TRAIN FUNCTION (FIXED LABELS)
# -----------------------------
def train_model(train_dataset, val_dataset, num_labels, id2label, label2id):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        id2label=id2label,     # ✅ FIX
        label2id=label2id      # ✅ FIX
    )

    model.to(device)

    training_args = TrainingArguments(
        output_dir="./intent_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,

        evaluation_strategy="steps",
        eval_steps=100,

        save_strategy="steps",
        save_steps=100,

        load_best_model_at_end=True,

        logging_dir="./logs",
        logging_steps=50,

        fp16=torch.cuda.is_available()
    )

    loss_tracker = LossTrackerCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            loss_tracker
        ]
    )

    trainer.train()

    return model, tokenizer, trainer, loss_tracker


# -----------------------------
# ROC-AUC
# -----------------------------
def compute_roc_auc(model, dataset, device, num_labels):

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataset:

            input_ids = batch["input_ids"].unsqueeze(0).to(device)
            attention_mask = batch["attention_mask"].unsqueeze(0).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

            all_probs.append(probs)
            all_labels.append(batch["label"].item())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    one_hot_labels = np.eye(num_labels)[all_labels]

    return roc_auc_score(one_hot_labels, all_probs, average="weighted", multi_class="ovr")


# -----------------------------
# LEARNING CURVE
# -----------------------------
def learning_curve_pipeline(dataset, num_labels, id2label, label2id):

    TRAIN_SIZES = [100, 300, 600, 1000, 2000]

    train_scores = []
    val_scores = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for size in TRAIN_SIZES:

        print(f"\nTraining with {size} samples...")

        subset = dataset["train"].select(range(min(size, len(dataset["train"]))))

        model, _, _, _ = train_model(
            subset,
            dataset["validation"],
            num_labels,
            id2label,
            label2id
        )

        train_auc = compute_roc_auc(model, subset, device, num_labels)
        val_auc = compute_roc_auc(model, dataset["validation"], device, num_labels)

        train_scores.append(train_auc)
        val_scores.append(val_auc)

    plt.figure()

    plt.plot(TRAIN_SIZES, train_scores, label="Train ROC-AUC")
    plt.plot(TRAIN_SIZES, val_scores, label="Validation ROC-AUC")

    plt.xlabel("Training Size")
    plt.ylabel("ROC-AUC")
    plt.title("Intent Model Learning Curve")

    plt.legend()
    plt.grid()

    plt.savefig("intent_learning_curve.png")
    plt.close()

    print("Saved learning curve -> intent_learning_curve.png")


# -----------------------------
# MAIN
# -----------------------------
def main():

    texts, labels = load_dataset()

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    #  CREATE LABEL MAPPINGS
    label2id = {label: i for i, label in enumerate(le.classes_)}
    id2label = {i: label for label, i in label2id.items()}

    num_labels = len(label2id)

    # SPLIT
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)

    dataset = DatasetDict({
        "train": Dataset.from_dict({"text": X_train, "label": y_train}),
        "validation": Dataset.from_dict({"text": X_val, "label": y_val}),
        "test": Dataset.from_dict({"text": X_test, "label": y_test})
    })

    dataset = dataset.map(tokenize)

    for split in dataset:
        dataset[split].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # TRAIN
    model, tokenizer, trainer, loss_tracker = train_model(
        dataset["train"],
        dataset["validation"],
        num_labels,
        id2label,
        label2id
    )

    # TEST EVAL
    print("\nFinal TEST evaluation:")
    print(trainer.evaluate(dataset["test"]))

    # SAVE
    SAVE_PATH = "models/intent_classifier_v4"

    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    # 🔥 SAVE LABEL MAPPING (VERY IMPORTANT)
    with open(f"{SAVE_PATH}/label_mapping.json", "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

    print(f"\nModel saved to {SAVE_PATH}")

    # LOSS CURVE
    plt.figure()
    plt.plot(loss_tracker.train_losses, label="Train Loss")
    plt.plot(loss_tracker.eval_losses, label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid()
    plt.savefig("intent_loss_curve.png")
    plt.close()

    print("Saved loss curve -> intent_loss_curve.png")

    # LEARNING CURVE
    learning_curve_pipeline(dataset, num_labels, id2label, label2id)


if __name__ == "__main__":
    main()