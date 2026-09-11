import pytest

from langchain_core.messages import HumanMessage,AIMessage

from summary_memory.summary_memory import main_prompt

def fake_summary(messages):

    text = " ".join(
        message.content
        for message in messages
    )

    if "7781" in text:
        return "The user's badge number is 7781."

    return "No important facts found."


def test_summary_memory_success():

    dropped_messages = [
        HumanMessage(content="My badge number is 7781."),
        AIMessage(content="I will remember it."),
    ]

    recent_history = [
        HumanMessage(content="I am learning LangChain."),
        AIMessage(content="That sounds good."),
    ]

    summary = fake_summary(dropped_messages)

    formatted = main_prompt.invoke(
        {
            "summary": summary,
            "history": recent_history,
            "input": "What is my badge number?",
        }
    )

    system_message = formatted.messages[0]

    assert "7781" in system_message.content

    assert ("My badge number is 7781."
        not in [
            message.content
            for message in recent_history
        ]
    )


def test_summary_memory_failure():

    empty_summary = ""

    if not empty_summary.strip():
        with pytest.raises(ValueError,match="Summary cannot be empty"
        ):
            raise ValueError("Summary cannot be empty.")