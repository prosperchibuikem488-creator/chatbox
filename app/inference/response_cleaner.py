import re

class ResponseCleaner:

    def clean(self, text):

        # Remove weird tokens
        text = text.replace("_comma_", ",")
        text = text.replace("_period_", ".")

        # Remove repetition
        sentences = text.split(".")
        seen = set()
        cleaned = []

        for s in sentences:
            s = s.strip()
            if s and s not in seen:
                cleaned.append(s)
                seen.add(s)

        return ". ".join(cleaned).strip()