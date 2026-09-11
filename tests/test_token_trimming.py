import pytest

from langchain_core.messages import HumanMessage, AIMessage

from token_trimming.token_trimming import (trim_history,simple_token_counter)


def test_token_trimming_success():

    messages = [
        HumanMessage(
            content="This is the first old message in the conversation."
        ),
        AIMessage(
            content="This is the first old response in the conversation."
        ),
        HumanMessage(
            content="This is another message that should use some tokens."
        ),
        AIMessage(
            content="This is another response that should use some tokens."
        ),
        HumanMessage(
            content="This is the newest human message."
        ),
        AIMessage(
            content="This is the newest AI response."
        ),
    ]

    trimmed = trim_history(messages,max_tokens=20)

    assert simple_token_counter(trimmed) <= 20

    assert trimmed[-1].content == (
        "This is the newest AI response."
    )

    for message in trimmed:
        assert message in messages


def test_token_trimming_failure():

    messages = [ HumanMessage(content="Hello")]

    with pytest.raises( ValueError, match="Token budget must be greater than 0"):
        trim_history(messages,max_tokens=0)