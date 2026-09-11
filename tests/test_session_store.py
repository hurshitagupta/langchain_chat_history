import pytest
from langchain_core.messages import HumanMessage

import session_store.session_store as store


def test_session_store_success(tmp_path, monkeypatch):

    test_db = f"sqlite:///{tmp_path}/test_history.db"

    monkeypatch.setattr(store,"DB_URL",test_db)

    history = store.get_history("test-user")

    history.add_message(
        HumanMessage(content="My favourite color is blue.")
    )

    # New object using same session ID
    reloaded_history = store.get_history("test-user")

    assert len(reloaded_history.messages) == 1
    assert (reloaded_history.messages[0].content == "My favourite color is blue.")


def test_session_store_failure():

    with pytest.raises(ValueError,match="Session ID cannot be empty"):
        store.get_history("")