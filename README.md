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
uv run python -m session_store.session_store > outputs/session_store.txt
```

Save the pytest output:

```bash
uv run pytest tests/test_session_store.py -v > outputs/test_session_store.txt
```

The saved artefacts provide reproducible evidence for the Task 1 deliverables.

## Task 1 Deliverables

Task 1 includes:

* Persistent SQLite session store
* History keyed by session ID
* Session ID validation
* Persistence demonstration
* Success and failure automated tests
* Saved execution output
* Saved pytest output
* Structured JSON trace

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
