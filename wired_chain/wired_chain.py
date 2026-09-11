import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openrouter import ChatOpenRouter

from session_store.session_store import get_history


load_dotenv()


def validate_input(user_input: str) -> None:
    if not user_input or not user_input.strip():
        raise ValueError("Input cannot be empty.")


model = ChatOpenRouter(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0,
    timeout=20_000,
    max_retries=3,
    max_tokens=100,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "Use the conversation history when answering."
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chain = prompt | model


conversational_chain = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)


def ask(session_id: str, user_input: str) -> str:
    validate_input(user_input)

    config = {
        "configurable": {
            "session_id": session_id
        }
    }

    response = conversational_chain.invoke(
        {"input": user_input},
        config=config
    )

    if not response.content.strip():
        raise ValueError("Model returned an empty response.")

    return response.content


if __name__ == "__main__":

    session_id = "wired-user-42"

    # Clear previous demo conversation
    get_history(session_id).clear()

    response1 = ask(
        session_id,
        "My badge number is 7781."
    )

    print("Turn 1:")
    print(response1)

    response2 = ask(
        session_id,
        "What is my badge number?"
    )

    print("\nTurn 2:")
    print(response2)

    history = get_history(session_id)

    print("\nStored history:")
    for message in history.messages:
        print(f"{message.type}: {message.content}")

    print(f"\nTotal messages stored: {len(history.messages)}")