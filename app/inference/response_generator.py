import json
import random
import os
from datetime import datetime

from dotenv import load_dotenv

from app.inference.emotion_predictor import EmotionPredictor
from app.inference.intent_predictor import IntentPredictor
from app.inference.llama_predictor import LlamaPredictor
from app.inference.safety_filter import SafetyFilter
from app.inference.response_cleaner import ResponseCleaner

load_dotenv()


class ResponseGenerator:

    def __init__(self):

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in .env file")

        self.emotion_model = EmotionPredictor()
        self.intent_model  = IntentPredictor()
        self.dialog_model  = LlamaPredictor(api_key=api_key)
        self.safety_filter = SafetyFilter()
        self.cleaner       = ResponseCleaner()

        with open("data/coping_strategies.json", "r", encoding="utf-8") as f:
            self.coping_strategies = json.load(f)

        self._reset_session()

    # ==========================================================
    # SESSION MANAGEMENT
    # ==========================================================

    def _reset_session(self):
        self.emotion_history      = []
        self.used_strategies      = {e: [] for e in self.coping_strategies}
        self.response_length_pref = None   # "brief" | "detailed" | None
        self.turn_count           = 0
        self.distress_score       = 0.0    # 0-1 rolling distress level

    # ==========================================================
    # PERSONALIZATION HELPERS
    # ==========================================================

    def _get_time_context(self):
         hour = datetime.now().hour
         if   5  <= hour < 12: return "morning"
         elif 12 <= hour < 17: return "afternoon"
         elif 17 <= hour < 21: return "evening"
         else:                 return "night"

    def _infer_length_preference(self, user_input):
        text = user_input.lower()
        if any(w in text for w in ["brief", "short", "quick", "tldr", "summarise", "summarize"]):
            return "brief"
        if any(w in text for w in ["explain", "tell me more", "in detail", "elaborate", "why"]):
            return "detailed"
        word_count = len(user_input.split())
        if word_count <= 5:  return "brief"
        if word_count >= 20: return "detailed"
        return None

    def _update_distress_score(self, primary_emotion):
        distress_map = {
            "sadness": 0.8, "anxiety": 0.7,
            "anger":   0.6, "neutral": 0.1, "joy": 0.0,
        }
        weight = distress_map.get(primary_emotion, 0.2)
        # Exponential moving average — recent emotions weight more
        self.distress_score = 0.6 * self.distress_score + 0.4 * weight

    def _pick_unused_strategy(self, emotion):
        available = self.coping_strategies.get(emotion, [])
        if not available:
            return None
        used   = self.used_strategies.get(emotion, [])
        unused = [s for s in available if s not in used]
        if not unused:
            self.used_strategies[emotion] = []
            unused = available
        chosen = random.choice(unused)
        self.used_strategies[emotion].append(chosen)
        return chosen

    def _should_show_strategy(self, primary_emotion, intent):
        if intent == "greeting" and self.turn_count <= 1:
            return False
        if primary_emotion in ("sadness", "anxiety", "anger"):
            return True
        if primary_emotion == "neutral" and self.turn_count % 2 == 0:
            return True
        if primary_emotion == "joy":
            return True
        return False

    # ==========================================================
    # MAIN GENERATE
    # ==========================================================

    def generate(self, user_input):

        self.turn_count += 1

        # -------------------------------------------------------
        # 1. SAFETY — check input BEFORE any processing
        # -------------------------------------------------------
        if self.safety_filter.check_crisis(user_input):
            return {
                "emotion": "crisis", "secondary_emotions": [],
                "intent": "crisis",
                "response": self.safety_filter.crisis_response()
            }

        if self.safety_filter.check_harm_others(user_input):
            return {
                "emotion": "anger", "secondary_emotions": [],
                "intent": "crisis",
                "response": self.safety_filter.harm_others_response()
            }

        if self.safety_filter.check_off_topic(user_input):
            return {
                "emotion": "neutral", "secondary_emotions": [],
                "intent": "off_topic",
                "response": self.safety_filter.off_topic_response()
            }

        # -------------------------------------------------------
        # 2. EMOTION DETECTION
        # -------------------------------------------------------
        emotions = self.emotion_model.predict_emotions(user_input)

        if emotions:
            sorted_emotions    = sorted(emotions, key=lambda x: x["confidence"], reverse=True)
            primary_emotion    = sorted_emotions[0]["emotion"]
            secondary_emotions = [e["emotion"] for e in sorted_emotions[1:3]]
        else:
            primary_emotion    = "neutral"
            secondary_emotions = []

        self.emotion_history.append(primary_emotion)
        self._update_distress_score(primary_emotion)

        # -------------------------------------------------------
        # 3. INTENT DETECTION
        # -------------------------------------------------------
        intent = self.intent_model.predict_intent(user_input)["intent"]

        # -------------------------------------------------------
        # 4. PERSONALIZATION SIGNALS
        # -------------------------------------------------------
        time_context  = self._get_time_context()

        inferred_pref = self._infer_length_preference(user_input)
        if inferred_pref:
            self.response_length_pref = inferred_pref
        length_pref = self.response_length_pref

        session_dominant = (
            max(set(self.emotion_history), key=self.emotion_history.count)
            if self.emotion_history else primary_emotion
        )

        # -------------------------------------------------------
        # 5. GENERATE RESPONSE (with personalization context)
        # -------------------------------------------------------
        response = self.dialog_model.generate_response(
            user_input               = user_input,
            primary_emotion          = primary_emotion,
            secondary_emotions       = secondary_emotions,
            intent                   = intent,
            time_context             = time_context,
            length_pref              = length_pref,
            session_dominant_emotion = session_dominant,
            distress_score           = self.distress_score,
        )

        # -------------------------------------------------------
        # 6. COPING STRATEGY (non-repetitive, context-aware)
        # -------------------------------------------------------
        if self._should_show_strategy(primary_emotion, intent):
            strategy = self._pick_unused_strategy(primary_emotion)
            if strategy:
                if time_context == "night":
                    prefix = "Before you sleep, you might find this helpful"
                elif time_context == "morning":
                    prefix = "To start your day well, you might find this helpful"
                else:
                    prefix = "You might find this helpful"
                response += f"\n\n{prefix}: {strategy}"

        # -------------------------------------------------------
        # 7. SAFETY FILTER ON OUTPUT
        # -------------------------------------------------------
        response = self.safety_filter.filter_response(user_input, response)

        # -------------------------------------------------------
        # 8. CLEAN
        # -------------------------------------------------------
        response = self.cleaner.clean(response)

        return {
            "emotion":            primary_emotion,
            "secondary_emotions": secondary_emotions,
            "intent":             intent,
            "response":           response
        }
