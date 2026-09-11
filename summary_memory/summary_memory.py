import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openrouter import ChatOpenRouter

from token_trimming.token_trimming import trim_history


load_dotenv()


model = ChatOpenRouter(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0,
    timeout=20_000,
    max_retries=3,
    max_tokens=150,
)


summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Summarise the conversation briefly. "
            "Keep important facts such as names, numbers, preferences, "
            "and personal details."
        ),
        (
            "human","{conversation}"
        ),
    ]
)


summary_chain = (summary_prompt | model | StrOutputParser())


main_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.\n"
            "Known summary from older conversation:\n{summary}"
        ),
        MessagesPlaceholder(variable_name="history"),
        (
            "human","{input}"
        ),
    ]
)


main_chain = (main_prompt | model | StrOutputParser())


def summarize_dropped_messages(messages) -> str:

    if not messages:
        return "(none)"

    conversation = "\n".join(
        f"{message.type}: {message.content}"
        for message in messages
    )

    summary = summary_chain.invoke(
        {"conversation": conversation}
    )

    if not summary.strip():
        raise ValueError("Summary cannot be empty.")

    return summary


def prepare_memory(messages, max_tokens: int):

    trimmed_messages = trim_history(messages,max_tokens=max_tokens)

    dropped_count = (len(messages) - len(trimmed_messages))

    dropped_messages = messages[:dropped_count]

    summary = summarize_dropped_messages(dropped_messages)

    return summary, trimmed_messages, dropped_messages


if __name__ == "__main__":

    messages = [
        HumanMessage(content="My badge number is 7781."),
        AIMessage(content="I will remember your badge number."),
        HumanMessage(content="I am currently learning LangChain."),
        AIMessage(content="That is a useful framework to learn."),
        HumanMessage(content="I am learning about chat history."),
        AIMessage(content="Chat history helps maintain conversation context."),
        HumanMessage(content="Now I am learning token trimming."),
        AIMessage(content="Token trimming keeps history within a budget."),
        HumanMessage(content="I am also learning summary memory."),
        AIMessage(content="Summary memory can preserve older information."),
    ]

    MAX_HISTORY_TOKENS = 25

    summary, recent_history, dropped = prepare_memory(messages,MAX_HISTORY_TOKENS)

    print(f"Dropped messages: {len(dropped)}")

    print(f"Recent messages kept: {len(recent_history)}")

    print("\nSummary:")
    print(summary)

    print("\nRecent history:")

    for message in recent_history:
        print(
            f"{message.type}: {message.content}"
        )

    question = "What is my badge number?"

    answer = main_chain.invoke(
        {
            "summary": summary,
            "history": recent_history,
            "input": question,
        }
    )

    if not answer.strip():
        raise ValueError("Model returned an empty response.")

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)