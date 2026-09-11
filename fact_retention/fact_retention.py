from langchain_core.messages import HumanMessage, AIMessage

from summary_memory.summary_memory import prepare_memory,main_chain

MAX_TURNS = 20
MAX_HISTORY_TOKENS = 40

def validate_turn_count(turn_count: int) -> None:
    if turn_count < 20:
        raise ValueError("Fact retention test requires at least 20 turns.")

def create_conversation(turn_count: int = MAX_TURNS):

    validate_turn_count(turn_count)

    messages = [
        HumanMessage(content="My badge number is 7781."),
        AIMessage(content="I will remember your badge number.")
    ]

    for turn in range(1, turn_count + 1):

        messages.append(HumanMessage(content=(f"Turn {turn}: Tell me one short fact about LangChain.")))

        messages.append(AIMessage(content=(f"Turn {turn}: LangChain helps build applications using language models.")))

    return messages


def run_fact_retention_test():

    messages = create_conversation()

    summary, recent_history, dropped_messages = prepare_memory(messages,MAX_HISTORY_TOKENS)

    original_fact = "My badge number is 7781."

    fact_in_recent_history = any(
        original_fact in message.content
        for message in recent_history
    )

    print(f"Later turns added: {MAX_TURNS}")
    print(f"Total messages: {len(messages)}")
    print(f"Dropped messages: {len(dropped_messages)}")
    print(f"Recent messages kept: {len(recent_history)}")

    print("Original fact still in recent history:",fact_in_recent_history)

    print("\nSummary:")
    print(summary)

    question = "What is my badge number?"

    answer = main_chain.invoke(
        {
            "summary": summary,
            "history": recent_history,
            "input": question,
        }
    )

    if not answer or not answer.strip():
        raise ValueError(
            "Model returned an empty response."
        )

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)

    return answer


if __name__ == "__main__":
    run_fact_retention_test()