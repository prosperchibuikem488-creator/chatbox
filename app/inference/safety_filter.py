import re


class SafetyFilter:

    def __init__(self):

        self.crisis_keywords = [
            "suicide", "kill myself", "end my life", "take my life",
            "i want to die", "i can't go on", "i cant go on",
            "self harm", "self-harm", "hurt myself", "cutting myself",
            "no reason to live", "not worth living", "better off dead",
            "want to disappear", "end it all", "overdose"
        ]

        self.harm_others_keywords = [
            "kill someone", "hurt someone", "hurt them",
            "attack", "stab", "shoot someone", "murder"
        ]

        self.profanity_list = [
            "fuck", "shit", "bitch", "asshole", "bastard",
            "damn", "crap", "piss", "dick", "cunt"
        ]

        self.off_topic_patterns = [
            r"\b(write me (a |some )?(code|script|program|essay|email))\b",
            r"\b(who (is|was) (the )?(president|prime minister|ceo))\b",
            r"\b(what is the (weather|stock|price|capital))\b",
            r"\b(translate (this|to|from))\b",
            r"\b(math|calculate|solve|equation)\b",
            r"\b(recipe|how (do i|to) cook)\b",
        ]

        self.unsafe_response_patterns = [
            r"\b(you should (harm|kill|hurt))\b",
            r"\b(here'?s how to (hurt|harm|kill))\b",
            r"\b(instructions (for|on) (harm|self.harm|suicide))\b",
        ]

    # -------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------

    def _contains(self, text, keyword_list):
        text_lower = text.lower()
        return any(kw in text_lower for kw in keyword_list)

    def _matches_pattern(self, text, patterns):
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in patterns)

    def check_crisis(self, text):
        return self._contains(text, self.crisis_keywords)

    def check_harm_others(self, text):
        return self._contains(text, self.harm_others_keywords)

    def check_off_topic(self, text):
        return self._matches_pattern(text, self.off_topic_patterns)

    def _contains_profanity(self, text):
        return self._contains(text, self.profanity_list)

    def _check_unsafe_response(self, text):
        return self._matches_pattern(text, self.unsafe_response_patterns)

    # -------------------------------------------------------
    # Pre-built safe responses
    # -------------------------------------------------------

    def crisis_response(self):
        return (
            "I can hear that you're in a lot of pain right now, and I'm really glad you're talking. "
            "You're not alone — what you're feeling matters, and so do you. \n\n"
            "Please consider reaching out to someone who can truly support you:\n"
            "• Talk to a trusted friend, family member, or someone close to you.\n"
            "• Contact a mental health professional or your nearest health centre.\n"
            "• If you're in immediate danger, please call your local emergency number.\n\n"
            "You deserve real support right now. I'm here to listen, but please reach out to someone who can help."
        )

    def harm_others_response(self):
        return (
            "It sounds like you're feeling a lot of anger or frustration right now — those feelings are real and valid. "
            "However, I'm not able to engage with thoughts about harming others. "
            "If things feel out of control, please speak with a counsellor or a trusted person who can help you work through this safely."
        )

    def off_topic_response(self):
        return (
            "I'm here specifically to support your emotional wellbeing and mental wellness. "
            "I'm not really set up to help with that kind of request, but I'm happy to listen if there's "
            "something on your mind or something you'd like to talk through. How are you feeling today?"
        )

    def _profanity_prefix(self):
        return (
            "It sounds like you might be feeling really frustrated or overwhelmed right now — "
            "that's completely okay. I'm here. \n\n"
        )

    # -------------------------------------------------------
    # Sanitize bot output
    # -------------------------------------------------------

    def _sanitize_response(self, text):
        text = re.sub(
            r"\b(harm|kill|die|murder|suicide|self.harm)\b",
            "",
            text,
            flags=re.IGNORECASE
        )
        return re.sub(r"  +", " ", text).strip()

    # -------------------------------------------------------
    # Main filter — called by response_generator.py
    # -------------------------------------------------------

    def filter_response(self, user_input, generated_response):

        # 1. Crisis — highest priority
        if self.check_crisis(user_input):
            return self.crisis_response()

        # 2. Harm to others
        if self.check_harm_others(user_input):
            return self.harm_others_response()

        # 3. Off-topic
        if self.check_off_topic(user_input):
            return self.off_topic_response()

        # 4. Profanity in user input — de-escalate, don't block
        if self._contains_profanity(user_input):
            clean = self._sanitize_response(generated_response)
            return self._profanity_prefix() + clean

        # 5. Unsafe content in bot response
        if self._check_unsafe_response(generated_response):
            return (
                "I want to make sure I'm being helpful and safe. "
                "I'm here to support you — can you tell me more about how you're feeling?"
            )

        # 6. General sanitize pass
        return self._sanitize_response(generated_response)
