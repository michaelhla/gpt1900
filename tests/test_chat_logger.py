"""Tests for the ChatLogger SQLite logging system."""

import os
import json
import tempfile
import pytest
from nanochat.chat_logger import ChatLogger


@pytest.fixture
def logger():
    """Create a ChatLogger with a temporary database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    lg = ChatLogger(path)
    yield lg
    lg.close()
    os.unlink(path)


def test_creates_table(logger):
    """Table exists after init."""
    rows = logger.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chat_logs'"
    ).fetchall()
    assert len(rows) == 1


def test_log_and_retrieve(logger):
    """Basic round-trip: log a request, read it back."""
    logger.log_request(
        conversation_id="conv-1",
        messages_json=json.dumps([{"role": "user", "content": "hello"}]),
        response_text="Hi there!",
        temperature=0.8,
        top_k=50,
        max_tokens=512,
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=123.4,
        client_ip="1.2.3.4",
        user_agent="test-agent",
        gpu_id=0,
    )

    logs = logger.get_logs(limit=10)
    assert len(logs) == 1

    row = logs[0]
    assert row["conversation_id"] == "conv-1"
    assert row["response_text"] == "Hi there!"
    assert row["temperature"] == 0.8
    assert row["prompt_tokens"] == 10
    assert row["completion_tokens"] == 5
    assert row["client_ip"] == "1.2.3.4"
    assert row["gpu_id"] == 0
    assert row["error"] is None
    assert row["timestamp"] is not None


def test_filter_by_conversation_id(logger):
    """Filtering by conversation_id returns only matching rows."""
    for cid in ["aaa", "bbb", "aaa"]:
        logger.log_request(
            conversation_id=cid,
            messages_json="[]",
            response_text="ok",
        )

    assert len(logger.get_logs(conversation_id="aaa")) == 2
    assert len(logger.get_logs(conversation_id="bbb")) == 1
    assert len(logger.get_logs(conversation_id="ccc")) == 0


def test_filter_by_time_range(logger):
    """since/until filters work on ISO timestamps."""
    logger.log_request(messages_json="[]", response_text="first")
    logs = logger.get_logs()
    ts = logs[0]["timestamp"]

    # "since" the exact timestamp should include it
    assert len(logger.get_logs(since=ts)) == 1
    # Far future should return nothing
    assert len(logger.get_logs(since="2099-01-01T00:00:00")) == 0


def test_limit_and_offset(logger):
    """Pagination works."""
    for i in range(10):
        logger.log_request(messages_json="[]", response_text=f"msg-{i}")

    page1 = logger.get_logs(limit=3, offset=0)
    page2 = logger.get_logs(limit=3, offset=3)
    assert len(page1) == 3
    assert len(page2) == 3
    # Results are DESC by id, so page1 has newer entries
    assert page1[0]["id"] > page2[0]["id"]


def test_error_logging(logger):
    """Error field is stored correctly."""
    logger.log_request(
        messages_json="[]",
        response_text="",
        error="CUDA out of memory",
    )
    logs = logger.get_logs()
    assert logs[0]["error"] == "CUDA out of memory"


def test_optional_fields_default_to_none(logger):
    """Only required fields; everything else is None."""
    logger.log_request(messages_json="[]", response_text="hi")
    row = logger.get_logs()[0]
    assert row["conversation_id"] is None
    assert row["temperature"] is None
    assert row["client_ip"] is None
    assert row["gpu_id"] is None
