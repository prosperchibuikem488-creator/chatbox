import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    hamming_loss
)


def compute_metrics(eval_pred):

    logits, labels = eval_pred

    # -----------------------------
    # Convert logits → probabilities
    # -----------------------------
    probs = 1 / (1 + np.exp(-logits))   # sigmoid

    # -----------------------------
    # Convert probabilities → predictions
    # -----------------------------
    threshold = 0.35
    preds = (probs >= threshold).astype(int)

    
    #  FIX: Convert soft labels → binary
   
    labels = (labels > 0).astype(int)

    # -----------------------------
    # Metrics
    # -----------------------------
    precision = precision_score(labels, preds, average="micro", zero_division=0)
    recall = recall_score(labels, preds, average="micro", zero_division=0)

    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    # ✅ Added: overall F1 (weighted gives balanced view)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    hamming = hamming_loss(labels, preds)

    return {
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "hamming_loss": hamming
    }