import re

class SafetyFilter:

    def __init__(self):
        self.crisis_keywords = [
            "suicide", "kill myself", "end my life",
            "i want to die", "i can't go on",
            "self harm", "hurt myself", "no reason to live"
        ]

    def check_crisis(self, text):
        text = text.lower()
        return any(keyword in text for keyword in self.crisis_keywords)

    def safe_response(self):
        return (
            "I'm really sorry you're feeling this way. "
            "You're not alone, and your life matters. "
            "It might really help to talk to someone you trust, like a friend or family member. "
            "If possible, please consider reaching out to a mental health professional or a support service near you."
        )

    def filter_response(self, user_input, generated_response):

        
        # 1. CRISIS OVERRIDE (CRITICAL)
        
        if self.check_crisis(user_input):
            return self.safe_response()

        
        # 2. CLEAN UNSAFE WORDS
        
        generated_response = re.sub(
            r"\b(harm|kill|die)\b",
            "",
            generated_response,
            flags=re.IGNORECASE
        )

        return generated_response.strip()