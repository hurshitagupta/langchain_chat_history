import json
import os
from datetime import datetime

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


DB_URL = os.getenv(
    "CHAT_HISTORY_DB_URL",
    "sqlite:///chat_history.db"
)


def validate_session_id(session_id: str) -> None:
    if not session_id or not session_id.strip():
        raise ValueError("Session ID cannot be empty.")


def get_history(session_id: str) -> BaseChatMessageHistory:
    validate_session_id(session_id)

    return SQLChatMessageHistory(
        session_id=session_id,
        connection=DB_URL
    )


def save_trace(session_id: str, message_count: int) -> None:
    os.makedirs("traces", exist_ok=True)

    trace = {
        "task": "session_store",
        "session_id": session_id,
        "message_count": message_count,
        "timestamp": datetime.now().isoformat()
    }

    with open(
        "traces/task1_session_store.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(trace, file, indent=2)


if __name__ == "__main__":

    session_id = "user-42"

    history = get_history(session_id)

    # Clear previous demo data so repeated runs stay predictable.
    history.clear()

    history.add_message(
        HumanMessage(content="My badge number is 7781.")
    )

    history.add_message(
        AIMessage(content="I will remember your badge number.")
    )

    print("Messages saved:")
    for message in history.messages:
        print(f"{message.type}: {message.content}")

    # Create a NEW history object to prove that
    # the messages came back from SQLite.
    reloaded_history = get_history(session_id)

    print("\nMessages loaded from SQLite:")
    for message in reloaded_history.messages:
        print(f"{message.type}: {message.content}")

    print(
        f"\nTotal persisted messages: "
        f"{len(reloaded_history.messages)}"
    )

    save_trace(
        session_id,
        len(reloaded_history.messages)
    )