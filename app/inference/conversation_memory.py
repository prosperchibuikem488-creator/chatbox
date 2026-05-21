class ConversationMemory:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history

    def add_turn(self, user, bot):
        self.history.append({"user": user, "bot": bot})

        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context(self):
        context = ""
        for turn in self.history:
            context += f"User: {turn['user']}\nBot: {turn['bot']}\n"
        return context