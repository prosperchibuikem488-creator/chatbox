import re
import requests


class LlamaPredictor:

    def __init__(self, api_key, max_history=10):
        self.api_key      = api_key
        self.url          = "https://api.mistral.ai/v1/chat/completions"
        self.chat_history = []
        self.max_history  = max_history

    # -----------------------------------
    # Clean response
    # -----------------------------------
    def clean_response(self, text):
        text = re.sub(r"\s+", " ", text).strip()
        if "### Response:" in text:
            text = text.split("### Response:")[-1].strip()
        if not text.endswith((".", "!", "?")):
            text += "."
        sentences = text.split(".")
        seen = []
        for s in sentences:
            s = s.strip()
            if s and s not in seen:
                seen.append(s)
        return ". ".join(seen).strip()

    # -----------------------------------
    # Build prompt - personalization-aware
    # -----------------------------------
    def build_prompt(
        self,
        user_input,
        primary_emotion,
        secondary_emotions       = None,
        intent                   = "general",
        time_context             = None,
        length_pref              = None,
        session_dominant_emotion = None,
        distress_score           = 0.0,
    ):
        history = ""
        for turn in self.chat_history[-self.max_history:]:
            history += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"

        tone_map = {
            "joy":     "Celebrate the user's positive feelings and reinforce happiness.",
            "sadness": "Be gentle, empathetic, and comforting.",
            "anxiety": "Be calming and reassuring. Help reduce worry.",
            "anger":   "Remain calm and help the user process anger constructively.",
            "neutral": "Be warm, friendly, and gently invite the user to share more.",
        }
        tone = tone_map.get(primary_emotion, "Be supportive, friendly, and conversational.")

        secondary_tone = ""
        if secondary_emotions:
            if "anxiety" in secondary_emotions: secondary_tone += " Add reassurance."
            if "sadness" in secondary_emotions: secondary_tone += " Show deeper empathy."
            if "anger"   in secondary_emotions: secondary_tone += " Avoid confrontation."
            if "joy"     in secondary_emotions: secondary_tone += " Reinforce positivity."

        intent_map = {
            "venting":        "Let the user express themselves freely. Focus on listening and validating. Do not rush to give advice.",
            "seeking_advice": "Provide gentle and practical coping suggestions without being prescriptive.",
            "crisis":         "Respond with urgency and deep empathy. Strongly encourage the user to seek real-world human support.",
            "greeting":       "Respond warmly, make the user feel welcome, and gently invite them to share how they are feeling.",
        }
        intent_instruction = intent_map.get(intent, "Provide a balanced and supportive response.")

        time_note = ""
        if time_context == "night":
            time_note = "It is late at night. Prefer calming, restful suggestions if recommending anything."
        elif time_context == "morning":
            time_note = "It is morning. You can encourage a positive, energised start to the day."
        elif time_context == "evening":
            time_note = "It is evening. Wind-down suggestions are appropriate if relevant."

        if length_pref == "brief":
            length_note = "Keep your response concise: 2 to 3 sentences maximum."
        elif length_pref == "detailed":
            length_note = "The user seems to want a fuller response. Aim for 5 to 6 thoughtful sentences."
        else:
            length_note = "Keep the response natural and human-like: 3 to 5 sentences."

        continuity_note = ""
        if session_dominant_emotion and session_dominant_emotion != primary_emotion:
            continuity_note = (
                f"Note: throughout this conversation the user has mostly been feeling "
                f"{session_dominant_emotion}. Keep that context in mind."
            )

        distress_note = ""
        if distress_score >= 0.65:
            distress_note = (
                "The user appears to have been in sustained emotional distress. "
                "Be especially gentle and consider suggesting professional support."
            )

        parts = [
            "You are a compassionate mental health support assistant.\n",
            "Core Rules:",
            "- Be empathetic, warm, and supportive at all times",
            "- Do NOT provide medical diagnosis, clinical advice, or treatment plans",
            "- Do NOT encourage harmful behaviour",
            "- Never repeat phrases already used in the conversation history",
            f"- {length_note}\n",
            f"Emotional Tone:\n{tone}{secondary_tone}\n",
            f"Intent Guidance:\n{intent_instruction}\n",
        ]
        if time_note:
            parts.append(f"Time Context: {time_note}\n")
        if continuity_note:
            parts.append(f"{continuity_note}\n")
        if distress_note:
            parts.append(f"{distress_note}\n")
        parts.append(f"Conversation History:\n{history if history else '(Start of conversation)'}\n")
        parts.append(f"User: {user_input}\n\nAssistant:")

        return "\n".join(parts)

    # -----------------------------------
    # Generate response
    # -----------------------------------
    def generate_response(
        self,
        user_input,
        primary_emotion          = "neutral",
        secondary_emotions       = None,
        intent                   = "general",
        time_context             = None,
        length_pref              = None,
        session_dominant_emotion = None,
        distress_score           = 0.0,
    ):
        prompt = self.build_prompt(
            user_input               = user_input,
            primary_emotion          = primary_emotion,
            secondary_emotions       = secondary_emotions,
            intent                   = intent,
            time_context             = time_context,
            length_pref              = length_pref,
            session_dominant_emotion = session_dominant_emotion,
            distress_score           = distress_score,
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

        payload = {
            "model":       "mistral-small-latest",
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens":  300,
        }

        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=15)
            result   = response.json()

            if "error" in result:
                print(f"[Mistral API Error] {result['error']}")
                generated_text = ""
            elif not result.get("choices"):
                print(f"[Mistral API] Unexpected response: {result}")
                generated_text = ""
            else:
                generated_text = result["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            print("[Mistral API] Request timed out.")
            generated_text = ""
        except Exception as e:
            print(f"[Mistral API] Unexpected error: {e}")
            generated_text = ""

        response = self.clean_response(generated_text)

        if not generated_text.strip() or len(response.split()) < 5:
            fallbacks = {
                "joy":     "That's wonderful to hear! I'm glad you're feeling good. Tell me more!",
                "sadness": "I'm really sorry you're feeling this way. I'm here for you — want to talk about it?",
                "anxiety": "It sounds like things feel overwhelming right now. Take a breath — I'm here with you.",
                "anger":   "It's okay to feel frustrated. I'm listening. What's been going on?",
            }
            response = fallbacks.get(
                primary_emotion,
                "I'm here and happy to chat. What's on your mind today?"
            )

        self.chat_history.append({"user": user_input, "bot": response})
        return response
