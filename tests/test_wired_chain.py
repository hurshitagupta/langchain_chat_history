import pytest

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory


def get_test_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///test_wired_history.db"
    )


def fake_model(data):
    history = data.get("history", [])

    if history:
        return AIMessage(content="I remember the previous conversation.")

    return AIMessage(content="This is the first message.")


fake_chain = RunnableLambda(fake_model)


test_chain = RunnableWithMessageHistory(
    fake_chain,
    get_test_history,
    input_messages_key="input",
    history_messages_key="history",
)


def test_wired_chain_success():

    session_id = "test-user"

    get_test_history(session_id).clear()

    config = {
        "configurable": {
            "session_id": session_id
        }
    }

    test_chain.invoke(
        {"input": "My badge number is 7781."},
        config=config
    )

    result = test_chain.invoke(
        {"input": "Do you remember me?"},
        config=config
    )

    assert result.content == "I remember the previous conversation."

    history = get_test_history(session_id)

    assert len(history.messages) == 4


def test_wired_chain_failure():

    with pytest.raises(ValueError):
        from wired_chain.wired_chain import validate_input

        validate_input("")