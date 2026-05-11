import re
import requests
from dotenv import load_dotenv
import os

load_dotenv()


class LlamaPredictor:

    def __init__(self, api_key, max_history=10):

        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.url = "https://api.mistral.ai/v1/chat/completions"

        # Conversation memory
        self.chat_history = []
        self.max_history = max_history


    # -----------------------------------
    # Clean response (UNCHANGED)
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
    # Build prompt (UNCHANGED)
    # -----------------------------------
    def build_prompt(self, user_input, primary_emotion, secondary_emotions=None, intent="general"):

        history = ""

        for turn in self.chat_history[-self.max_history:]:
            history += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"

        if primary_emotion == "joy":
            tone = "Celebrate the user's positive feelings and reinforce happiness."
        elif primary_emotion == "sadness":
            tone = "Be gentle, empathetic, and comforting."
        elif primary_emotion == "anxiety":
            tone = "Be calming and reassuring. Reduce worry."
        elif primary_emotion == "anger":
            tone = "Remain calm and help the user process anger constructively."
        else:
            tone = "Be supportive, friendly, and conversational."

        secondary_tone = ""

        if secondary_emotions:
            if "anxiety" in secondary_emotions:
                secondary_tone += " Add reassurance."
            if "sadness" in secondary_emotions:
                secondary_tone += " Show deeper empathy."
            if "anger" in secondary_emotions:
                secondary_tone += " Avoid confrontation."
            if "joy" in secondary_emotions:
                secondary_tone += " Reinforce positivity."

        if intent == "venting":
            intent_instruction = "Let the user express themselves. Focus on listening and validating feelings."
        elif intent == "seeking_advice":
            intent_instruction = "Provide gentle and practical coping suggestions."
        elif intent == "crisis":
            intent_instruction = "Respond with urgency, empathy, and encourage seeking real-world help."
        elif intent == "greeting":
            intent_instruction = "Respond warmly and invite conversation."
        else:
            intent_instruction = "Provide a balanced supportive response."

        prompt = f"""You are a mental health support assistant.

Rules:
- Be empathetic and supportive 
- Do NOT provide medical or clinical advice
- Do NOT encourage harmful behavior
- Keep responses natural and human-like (3–6 sentences)

Emotion Tone:
{tone}
{secondary_tone}

Intent Guidance:
{intent_instruction}

Conversation History:
{history}

User: {user_input}

Assistant:
"""

        return prompt


    # -----------------------------------
    # Generate response (API VERSION)
    # -----------------------------------
    def generate_response(
        self,
        user_input,
        primary_emotion="neutral",
        secondary_emotions=None,
        intent="general"
    ):

        prompt = self.build_prompt(
            user_input,
            primary_emotion,
            secondary_emotions,
            intent
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mistral-small",   # or "mistral-medium"
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 200
        }

        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=15)
            result = response.json()

            # Check for API-level errors (e.g. auth failure, bad request)
            if "error" in result:
                print(f"[Mistral API Error] {result['error']}")
                generated_text = ""

            # Guard against empty or missing choices
            elif not result.get("choices"):
                print(f"[Mistral API] Unexpected response format: {result}")
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

        # Fallback — covers empty string (API error) AND too-short responses
        if not generated_text.strip() or len(response.split()) < 5:
            fallbacks = {
                "joy": "That's wonderful to hear! I'm glad you're feeling good. Tell me more!",
                "sadness": "I'm really sorry you're feeling this way. I'm here for you — want to talk about it?",
                "anxiety": "It sounds like things feel overwhelming right now. Take a breath — I'm here with you.",
                "anger": "It's okay to feel frustrated. I'm listening. What's been going on?",
            }
            response = fallbacks.get(
                primary_emotion,
                "I'm here and happy to chat. What's on your mind today?"
            )

        # Save memory
        self.chat_history.append({
            "user": user_input,
            "bot": response
        })

        return response