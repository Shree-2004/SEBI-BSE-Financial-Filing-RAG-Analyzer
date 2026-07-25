"""
logging_config.py

Structured JSON logging for the API, replacing raw print() statements.
Right now, debugging a production issue means grepping unstructured prints
with no way to correlate a specific request to a specific failure. JSON logs
mean each line is one queryable event, and hosts like Railway/Render (which
capture stdout) can parse them directly.

Deliberately stdlib-only (no new dependency) — a custom Formatter is enough
for this project's scale.

Usage:
    from api.logging_config import setup_logging, get_logger

    setup_logging()                 # once, at app startup
    logger = get_logger(__name__)
    logger.info("query completed", extra={"request_id": rid, "latency_ms": 42})
"""

import json
import logging
import os
import sys
import time

# Standard LogRecord attributes — anything else on the record came from
# `extra={...}` and should be surfaced as a structured field.
_RESERVED_KEYS = set(logging.LogRecord(
    "", 0, "", 0, "", (), None
).__dict__.keys()) | {"message", "asctime"}


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED_KEYS}
        payload.update(extras)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # ensure_ascii avoids any risk of a non-ASCII character crashing on
        # a Windows console using a non-UTF-8 codepage, same failure mode
        # hit earlier in this project with print() + unicode checkmarks.
        return json.dumps(payload, ensure_ascii=True)


def setup_logging() -> None:
    """Configure the root logger once, at app startup. Idempotent — safe
    to call multiple times (e.g. under uvicorn's --reload)."""
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root.setLevel(level)
    root.addHandler(handler)

    # uvicorn's own loggers otherwise use their own plain-text format —
    # route them through the same JSON formatter for consistency.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers = [handler]
        uv_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
