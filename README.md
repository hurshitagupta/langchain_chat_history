# LangChain Chat History

Hands-on assessment focused on implementing persistent chat history in LangChain while keeping conversation memory within a controlled token budget.

The assessment covers session-based history storage, `RunnableWithMessageHistory`, token trimming, summary memory, and long-term fact retention.

---

# Task 1 — Session Store

## Objective

Implement a persistent chat history store where conversations are separated using a unique session ID.

SQLite is used as the persistent backend through LangChain's `SQLChatMessageHistory`.

## Implementation

The session history is created using:

```python
def get_history(session_id: str) -> BaseChatMessageHistory:
    validate_session_id(session_id)

    return SQLChatMessageHistory(
        session_id=session_id,
        connection=DB_URL
    )
```

Each conversation is identified using a `session_id`.

The SQLite database is stored locally as:

```text
chat_history.db
```

This allows conversation history to remain available even after a new history object is created.

## Validation

Session IDs are validated before accessing the history:

```python
def validate_session_id(session_id: str) -> None:
    if not session_id or not session_id.strip():
        raise ValueError("Session ID cannot be empty.")
```

An empty session ID is rejected instead of creating an invalid history session.

## Persistence Test

The script first stores messages in the session:

```text
human: My badge number is 7781.
ai: I will remember your badge number.
```

A new `SQLChatMessageHistory` instance is then created using the same session ID.

The messages are loaded again from SQLite, demonstrating that the history is persisted rather than existing only in memory.

## Structured Trace

Task 1 also creates a structured JSON trace:

```text
traces/task1_session_store.json
```

## Run Task 1

```bash
uv run python -m session_store.session_store
```

## Tests

Task 1 contains two automated tests:

* **Success case:** stores a message and verifies that it can be retrieved using a new history instance with the same session ID.
* **Failure case:** verifies that an empty session ID raises a `ValueError`.

Run the tests using:

```bash
uv run pytest tests/test_session_store.py -v
```

## Save Evidence

Save the Task 1 execution output:

```bash
uv run python -m session_store.session_store > outputs/session_store_output.txt
```

Save the pytest output:

```bash
uv run pytest tests/test_session_store.py -v > outputs/test_session_store.txt
```

The saved artefacts provide reproducible evidence for the Task 1 deliverables.

---

## Guardrails

Guardrails will be implemented throughout the assessment where applicable.

Task 1 does not make an LLM/API call, so model-specific controls such as timeout and retry are not applicable at this stage.

Currently implemented:

* **Input validation** — empty session IDs are rejected.
* **Persistent storage** — conversation history is stored in SQLite rather than temporary memory.
* **Structured tracing** — session activity and message counts are recorded in JSON.
* **Secret hygiene** — no API keys or credentials are stored in the source code.

Additional controls including step limits, timeout, retry, token budgets, and model-output validation will be introduced when model calls are added in the following tasks.

---

# Task 2 — Wired Chain

## Objective

Use `RunnableWithMessageHistory` so previous conversation messages are automatically retrieved and injected into the chain based on the session ID.

The persistent SQLite session store created in Task 1 is reused for this task.

## Implementation

The prompt contains a `MessagesPlaceholder` for the conversation history:

```python id="5azjpt"
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
```

The prompt and model are combined into the normal chain:

```python id="g0byh3"
chain = prompt | model
```

The chain is then wrapped using `RunnableWithMessageHistory`:

```python id="19vvne"
conversational_chain = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)
```

The `get_history` function from Task 1 provides the SQLite history for the given session.

The session ID is passed through the runnable configuration:

```python id="4ac9c1"
config = {
    "configurable": {
        "session_id": session_id
    }
}
```

This allows different session IDs to maintain separate conversation histories.

## Automatic History

Unlike Task 1, messages are not manually added to the history.

The first interaction stores the user's message and the model response automatically.


## Input and Output Validation

Empty user input is rejected before invoking the model:

```python id="y4vwkf"
def validate_input(user_input: str) -> None:
    if not user_input or not user_input.strip():
        raise ValueError("Input cannot be empty.")
```

The returned model response is also checked:

```python id="7a19c0"
if not response.content.strip():
    raise ValueError("Model returned an empty response.")
```

This prevents an empty model response from being returned to the caller.

## Guardrails

Implemented controls include:

* **Timeout** — limits how long a model request can wait.
* **Retry** — model requests use capped retries.
* **Output token limit** — limits the maximum model response size.
* **Input validation** — empty user input is rejected.
* **Output validation** — empty model responses are rejected.
* **Secret hygiene** — API configuration is loaded from environment variables rather than hard-coded into the source.

Conversation-history token trimming is implemented separately in Task 3.

## Run Task 2

Run the wired chain using:

```bash id="xeyy99"
uv run python -m wired_chain.wired_chain
```

## Tests

Task 2 includes two automated tests.

### Success Case

A deterministic fake runnable is wrapped with `RunnableWithMessageHistory`.

Two interactions are performed using the same session ID. The test verifies that previous history is available during the second interaction and that four messages are stored after two turns.

### Failure Case

The failure test passes an empty input and verifies that the input validation raises a `ValueError`.

Run the tests using:

```bash id="pxrjzm"
uv run pytest tests/test_wired_chain.py -v
```

## Save Evidence

Save the script output:

```bash id="qfoc1b"
uv run python -m wired_chain.wired_chain > outputs/wired_chain_output.txt
```

Save the test output:

```bash id="y1wm0a"
uv run pytest tests/test_wired_chain.py -v > outputs/test_wired_chain.txt
```
---

# Task 3 — Token Trimming

## Objective

Implement a token trimming policy so that long conversation histories remain within a defined token budget.

The trimming keeps the most recent messages and removes older messages on message boundaries instead of splitting individual messages.

## Implementation

A fixed token budget is defined:

```python id="42lhxk"
MAX_TOKENS = 40
```

For this task, a simple deterministic token counter is used:

```python id="2axrxe"
def simple_token_counter(messages) -> int:
    return sum(
        len(message.content.split())
        for message in messages
    )
```

Each whitespace-separated word is counted as one token. This provides a predictable measurement for demonstrating and testing the trimming behaviour.

The conversation is trimmed using LangChain's `trim_messages` utility:

```python id="j2c7vy"
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
```

The `last` strategy keeps the most recent conversation messages when the complete history exceeds the available token budget.

Using:

```python id="fp9y9q"
start_on="human"
```

also helps ensure that the retained conversation begins with a human message rather than an isolated AI response.

## Message Boundary Trimming

Messages are removed as complete message objects.

The implementation does not truncate message content manually.

For example, the code does not perform operations such as:

```python id="xqezpt"
message.content[:100]
```

Instead, a message is either retained completely or removed from the context.

This satisfies the requirement that trimming must occur on message boundaries.

## Token Measurement

Token usage is measured before and after trimming:

```python id="jhbce4"
before_tokens = simple_token_counter(messages)

trimmed_messages = trim_history(messages)

after_tokens = simple_token_counter(trimmed_messages)
```

The script reports:

* configured token budget
* token count before trimming
* token count after trimming
* message count before trimming
* message count after trimming

This provides measurable evidence that the trimming policy is working.

## Run Task 3

Run the token trimming implementation using:

```bash id="dxz6pe"
uv run python -m token_trimming.token_trimming
```

## Tests

Task 3 contains automated success and failure tests.

### Success Case

The success test creates a conversation that exceeds a small token budget and verifies that:

* the final token count is within the configured budget
* the most recent message remains available
* retained messages are complete original message objects

### Failure Case

The implementation rejects the invalid configuration by raising a `ValueError`.

Run the tests using:

```bash id="zmfj7s"
uv run pytest tests/test_token_trimming.py -v
```

## Save Evidence

Save the Task 3 execution output:

```bash id="d5o9ji"
uv run python -m token_trimming.token_trimming > outputs/token_trimming_output.txt
```

Save the pytest output:

```bash id="8axjbg"
uv run pytest tests/test_token_trimming.py -v > outputs/test_token_trimming.txt
```

## Guardrails

The following controls are demonstrated in Task 3:

* **Token budget** — conversation history is restricted to a defined maximum token count.
* **Input/configuration validation** — zero or negative token budgets are rejected.
* **Message-boundary trimming** — complete messages are retained or removed rather than partially truncated.
* **Deterministic measurement** — token counts before and after trimming are reported numerically.

Model-specific timeout and retry controls remain part of the model configuration introduced in Task 2.
