import json
import os
import random
import re



# Load emotion mapping

def load_mapping():
    mapping_path = os.path.join("data", "emotion_mapping.json")
    with open(mapping_path, "r") as f:
        return json.load(f)



# Load label schema

def load_label_schema():
    label_path = os.path.join("data", "labels.json")
    with open(label_path, "r") as f:
        labels_data = json.load(f)

    id2label = labels_data["id2label"]
    label_schema = [id2label[str(i)] for i in range(len(id2label))]

    return label_schema



# Normalize text

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"http\S+", "", text)
    text = text.strip()
    return text



# Clean GoEmotions labels

def clean_goemotions_labels(labels, mapping_dict):
    labels = [l for l in labels if l in mapping_dict]

    # Only remove completely invalid samples
    if len(labels) == 0:
        return None

    return labels



# Encode labels (multi-label safe)

def encode_labels(original_labels, mapping_dict, label_schema):

    vector = [0.0] * len(label_schema)
    label2id = {label: i for i, label in enumerate(label_schema)}

    # accumulate signal (better for multi-label)
    for orig in original_labels:
        mapped_label = mapping_dict[orig]

        if mapped_label in label2id:
            index = label2id[mapped_label]
            vector[index] += 1.0

    # normalize to [0,1]
    max_val = max(vector)
    if max_val > 0:
        vector = [v / max_val for v in vector]

    return vector



# Light synonym augmentation

synonym_dict = {
    "happy": ["joyful", "glad"],
    "sad": ["down", "depressed"],
    "angry": ["mad", "furious"],
    "anxious": ["nervous", "worried"]
}


def synonym_augmentation(text):
    words = text.split()
    new_words = []

    for word in words:
        clean_word = re.sub(r"[^\w]", "", word)

        if clean_word in synonym_dict and random.random() < 0.1:
            new_words.append(random.choice(synonym_dict[clean_word]))
        else:
            new_words.append(word)

    return " ".join(new_words)



# Main preprocessing pipeline

def preprocess_dataset(dataset, tokenizer):

    mapping = load_mapping()
    label_schema = load_label_schema()

    goemotion_names = dataset["train"].features["labels"].feature.names

   
    # Filter invalid samples
    
    def filter_invalid(example):
        original_labels = [goemotion_names[i] for i in example["labels"]]
        cleaned_labels = clean_goemotions_labels(original_labels, mapping)
        return cleaned_labels is not None

    filtered_dataset = dataset.filter(filter_invalid)

    
    # Tokenization + encoding
   
    def tokenize_and_encode(example):

        text = normalize_text(example["text"])

        # light augmentation (reduced)
        if random.random() < 0.1:
            text = synonym_augmentation(text)

        original_labels = [goemotion_names[i] for i in example["labels"]]
        cleaned_labels = clean_goemotions_labels(original_labels, mapping)

        # safety fallback
        if cleaned_labels is None:
            cleaned_labels = ["neutral"]

        tokenized = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_token_type_ids=False
        )

        tokenized["labels"] = encode_labels(
            cleaned_labels,
            mapping,
            label_schema
        )

        return tokenized

    encoded_dataset = filtered_dataset.map(
        tokenize_and_encode,
        remove_columns=dataset["train"].column_names
    )

    return encoded_dataset