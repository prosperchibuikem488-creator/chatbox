import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EmotionPredictor:

    def __init__(self, model_path="models/emotion_classifier_v6", threshold=0.30):

        self.model_path = model_path
        self.threshold = threshold

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Emotion Predictor running on:", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path
        )

        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label


    def predict_emotions(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probs = torch.sigmoid(logits)

        probs = probs[0].cpu().numpy()

        predicted_emotions = []

        # -----------------------------
        # THRESHOLD FILTERING
        # -----------------------------
        for i, prob in enumerate(probs):

            if prob >= self.threshold:
                predicted_emotions.append(
                    {
                        "emotion": self.id2label[i],
                        "confidence": float(prob)
                    }
                )

        # -----------------------------
        # FALLBACK (if empty)
        # -----------------------------
        if len(predicted_emotions) == 0:

            max_index = probs.argmax()

            predicted_emotions.append(
                {
                     "emotion": self.id2label[max_index],
                     "confidence": float(probs[max_index])
                }
            )

        # -----------------------------
        # SORT + LIMIT
        # -----------------------------
        predicted_emotions = sorted(
            predicted_emotions,
            key=lambda x: x["confidence"],
            reverse=True
        )

        predicted_emotions = predicted_emotions[:3]

        return predicted_emotions


# -----------------------------
# TESTING MAIN LOOP
# -----------------------------
if __name__ == "__main__":

    predictor = EmotionPredictor()

    print("\n=== Emotion Predictor Test ===")
    print("Type 'quit' to exit.\n")

    while True:

        text = input("Input: ").strip()

        if text.lower() == "quit":
            print("Exiting Emotion Predictor.\n")
            break

        if not text:
            continue

        emotions = predictor.predict_emotions(text)

        print("\nDetected Emotions:")

        for i, e in enumerate(emotions):
            label = "Primary" if i == 0 else "Secondary"
            print(f"{label}: {e['emotion']} ({e['confidence']:.2f})")

        print()