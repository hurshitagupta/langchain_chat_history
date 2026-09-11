import pytest

from fact_retention.fact_retention import create_conversation,validate_turn_count


def deterministic_summary(messages):

    for message in messages:
        if "7781" in message.content:
            return "The user's badge number is 7781."

    return "No important fact found."


def answer_from_summary(summary: str) -> str:

    if "7781" in summary:
        return "Your badge number is 7781."

    return "I do not know your badge number."


def test_fact_retention_success():

    messages = create_conversation(20)

    recent_history = messages[-6:]

    dropped_messages = messages[:-6]

    original_fact = "My badge number is 7781."

    assert all(
        original_fact not in message.content
        for message in recent_history
    )

    summary = deterministic_summary(dropped_messages)

    answer = answer_from_summary(summary)

    assert "7781" in summary
    assert "7781" in answer


def test_fact_retention_failure():

    with pytest.raises(ValueError,match="requires at least 20 turns"):
        validate_turn_count(10)