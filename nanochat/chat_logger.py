"""
Lightweight SQLite logger for chat request/response pairs.

Usage:
    logger = ChatLogger("chat_logs.db")
    logger.log_request(
        conversation_id="abc",
        messages_json='[{"role":"user","content":"hi"}]',
        response_text="Hello!",
        temperature=0.8,
        ...
    )
    logs = logger.get_logs(limit=50)
    logger.close()
"""

import sqlite3
import json
from datetime import datetime, timezone


class ChatLogger:
    def __init__(self, db_path: str = "chat_logs.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                conversation_id TEXT,
                messages_json TEXT NOT NULL,
                response_text TEXT NOT NULL,
                temperature REAL,
                top_k INTEGER,
                max_tokens INTEGER,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                latency_ms REAL,
                ttft_ms REAL,
                client_ip TEXT,
                user_agent TEXT,
                gpu_id INTEGER,
                error TEXT
            )
        """)
        # Indexes for common queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_conversation_id ON chat_logs(conversation_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON chat_logs(timestamp)")
        self.conn.commit()
        self._migrate()

    def _migrate(self):
        """Add columns that may be missing from older databases."""
        existing = {row[1] for row in self.conn.execute("PRAGMA table_info(chat_logs)")}
        if "ttft_ms" not in existing:
            self.conn.execute("ALTER TABLE chat_logs ADD COLUMN ttft_ms REAL")
            self.conn.commit()

    def log_request(self, *, conversation_id=None, messages_json="[]",
                    response_text="", temperature=None, top_k=None,
                    max_tokens=None, prompt_tokens=None, completion_tokens=None,
                    latency_ms=None, ttft_ms=None, client_ip=None, user_agent=None,
                    gpu_id=None, error=None):
        self.conn.execute(
            """INSERT INTO chat_logs
               (timestamp, conversation_id, messages_json, response_text,
                temperature, top_k, max_tokens, prompt_tokens, completion_tokens,
                latency_ms, ttft_ms, client_ip, user_agent, gpu_id, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                conversation_id,
                messages_json,
                response_text,
                temperature,
                top_k,
                max_tokens,
                prompt_tokens,
                completion_tokens,
                latency_ms,
                ttft_ms,
                client_ip,
                user_agent,
                gpu_id,
                error,
            ),
        )
        self.conn.commit()

    def get_logs(self, limit=50, offset=0, conversation_id=None, since=None, until=None):
        query = "SELECT * FROM chat_logs WHERE 1=1"
        params = []

        if conversation_id:
            query += " AND conversation_id = ?"
            params.append(conversation_id)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        if until:
            query += " AND timestamp <= ?"
            params.append(until)

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def close(self):
        self.conn.close()
