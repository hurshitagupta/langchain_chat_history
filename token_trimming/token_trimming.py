from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages.utils import trim_messages


MAX_TOKENS = 40


def simple_token_counter(messages) -> int:
    """
    Simple deterministic token counter for this assessment.
    Each whitespace-separated word is counted as one token.
    """
    return sum(
        len(message.content.split())
        for message in messages
    )


def trim_history(messages, max_tokens: int = MAX_TOKENS):

    if max_tokens <= 0:
        raise ValueError("Token budget must be greater than 0.")

    trimmed = trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy="last",
        token_counter=simple_token_counter,
        start_on="human",
    )

    return trimmed


if __name__ == "__main__":

    messages = [
        HumanMessage(
            content="My name is Hurshita and I am learning LangChain."
        ),
        AIMessage(
            content="Nice to meet you. I will remember that information."
        ),
        HumanMessage(
            content="I am currently learning how chat history works."
        ),
        AIMessage(
            content="Chat history allows models to use previous conversation messages."
        ),
        HumanMessage(
            content="Now I am learning how token trimming works."
        ),
        AIMessage(
            content="Token trimming keeps the conversation within a token budget."
        ),
    ]

    before_tokens = simple_token_counter(messages)

    trimmed_messages = trim_history(messages)

    after_tokens = simple_token_counter(trimmed_messages)

    print(f"Token budget: {MAX_TOKENS}")
    print(f"Tokens before trimming: {before_tokens}")
    print(f"Tokens after trimming: {after_tokens}")

    print(f"\nMessages before trimming: {len(messages)}")
    print(f"Messages after trimming: {len(trimmed_messages)}")

    print("\nTrimmed history:")

    for message in trimmed_messages:
        print(f"{message.type}: {message.content}")