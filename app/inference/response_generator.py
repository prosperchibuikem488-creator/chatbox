import json
import random
import os

from dotenv import load_dotenv

from inference.emotion_predictor import EmotionPredictor
from inference.intent_predictor import IntentPredictor
from inference.llama_predictor import LlamaPredictor
from inference.safety_filter import SafetyFilter
from inference.response_cleaner import ResponseCleaner


# Load environment variables
load_dotenv()


class ResponseGenerator:

    def __init__(self):

        # -----------------------------
        # LOAD API KEY
        # -----------------------------
        api_key = os.getenv("MISTRAL_API_KEY")

        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY not found in .env file"
            )

        # -----------------------------
        # LOAD MODELS
        # -----------------------------
        self.emotion_model = EmotionPredictor()
        self.intent_model = IntentPredictor()

        self.dialog_model = LlamaPredictor(
            api_key=api_key
        )

        self.safety_filter = SafetyFilter()
        self.cleaner = ResponseCleaner()

        # -----------------------------
        # LOAD COPING STRATEGIES
        # -----------------------------
        with open(
            "data/coping_strategies.json",
            "r",
            encoding="utf-8"
        ) as f:

            self.coping_strategies = json.load(f)


    def generate(self, user_input):

        # -----------------------------
        # 1. EMOTION DETECTION
        # -----------------------------
        emotions = self.emotion_model.predict_emotions(user_input)

        if emotions:

            sorted_emotions = sorted(
                emotions,
                key=lambda x: x["confidence"],
                reverse=True
            )

            primary_emotion = sorted_emotions[0]["emotion"]

            secondary_emotions = [
                e["emotion"] for e in sorted_emotions[1:3]
            ]

        else:

            primary_emotion = "neutral"
            secondary_emotions = []

        # -----------------------------
        # 2. INTENT DETECTION
        # -----------------------------
        intent_result = self.intent_model.predict_intent(
            user_input
        )

        intent = intent_result["intent"]

        # -----------------------------
        # 3. GENERATE RESPONSE
        # -----------------------------
        response = self.dialog_model.generate_response(
            user_input=user_input,
            primary_emotion=primary_emotion,
            secondary_emotions=secondary_emotions,
            intent=intent
        )

        # -----------------------------
        # 4. ADD COPING STRATEGY
        # -----------------------------
        if primary_emotion in self.coping_strategies:

            strategy = random.choice(
                self.coping_strategies[primary_emotion]
            )

            response += (
                f"\n\nYou might find this helpful: {strategy}"
            )

        # -----------------------------
        # 5. SAFETY FILTER
        # -----------------------------
        response = self.safety_filter.filter_response(
            user_input,
            response
        )

        # -----------------------------
        # 6. CLEAN RESPONSE
        # -----------------------------
        response = self.cleaner.clean(response)

        return {
            "emotion": primary_emotion,
            "secondary_emotions": secondary_emotions,
            "intent": intent,
            "response": response
        }