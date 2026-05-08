from inference.response_generator import ResponseGenerator


def main():

    print("\n=== Mental Health Chatbot (LLaMA Powered) ===")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    bot = ResponseGenerator()

    while True:

        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("\nBot: Take care of yourself. I'm here whenever you need support.\n")
                break

            if not user_input:
                continue

            result = bot.generate(user_input)

            # -----------------------------
            # ANALYSIS OUTPUT
            # -----------------------------
            print("\n--- Analysis ---")
            print(f"Primary Emotion : {result['emotion']}")

            # Secondary emotions (NEW)
            secondary = result.get("secondary_emotions", [])
            if secondary:
                print(f"Secondary Emotions : {', '.join(secondary)}")
            else:
                print("Secondary Emotions : None")

            print(f"Intent : {result['intent']}")

            # -----------------------------
            # RESPONSE OUTPUT
            # -----------------------------
            print("\n--- Response ---")
            print(f"{result['response']}\n")

        except KeyboardInterrupt:
            print("\n\nBot: Session ended. Take care!\n")
            break

        except Exception as e:
            print("\n[Error] Something went wrong:")
            print(str(e))
            print("Please try again.\n")


if __name__ == "__main__":
    main() 







