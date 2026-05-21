import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class IntentPredictor:

    def __init__(self, model_path="models/intent_classifier_v4"):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Intent Predictor running on:", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label


    def predict_intent(self, text):

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

        probs = torch.softmax(logits, dim=1)

        probs = probs.squeeze().cpu().numpy()

        predicted_index = probs.argmax()

        return {
            "intent": self.id2label[predicted_index],
            "confidence": float(probs[predicted_index])
        }


# -----------------------------
# Testing loop
# -----------------------------
if __name__ == "__main__":

    predictor = IntentPredictor()

    while True:

        text = input("\nEnter a message (or 'quit'): ")

        if text.lower() == "quit":
             break

        result = predictor.predict_intent(text)

        print("\nPredicted Intent:")
        print(f"{result['intent']} ({result['confidence']:.2f})")