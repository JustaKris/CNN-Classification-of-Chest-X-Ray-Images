"""Centralized logging configuration for both local development and production deployments.

Features:
- JSON logging for structured cloud logs (Azure, Docker, Kubernetes)
- Colorized human-readable logs for local development
- Suppression of noisy 3rd-party libraries
- Support for extra contextual fields (e.g., request_id)
"""

import json
import logging
import sys
from typing import Any


class JsonFormatter(logging.Formatter):
    """Formatter that outputs logs as structured JSON.

    Recommended for production — parseable by Azure Monitor, CloudWatch, Datadog, Loki, etc.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Convert log record into a JSON object."""
        log_record: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "process": record.process,
            "thread": record.thread,
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            log_record.update(extra_fields)

        return json.dumps(log_record)


class ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI colors to log levels for local development readability."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Inject color codes into the level name without mutating the original record."""
        original = record.levelname
        color = self.COLORS.get(original, "")
        record.levelname = f"{color}{original}{self.RESET}"

        formatted = super().format(record)

        record.levelname = original
        return formatted


def configure_logging(
    level: str = "INFO",
    use_json: bool = False,
) -> None:
    """Configure application-wide logging.

    Args:
        level: Default logging level (INFO, DEBUG, WARNING, ...).
        use_json: Enables JSON logs (recommended for production).
    """
    level = level.upper()

    formatter: logging.Formatter
    if use_json:
        formatter = JsonFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = ColoredFormatter(fmt=fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Suppress noisy 3rd-party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured: level=%s, format=%s",
        level,
        "JSON" if use_json else "colored",
    )


def get_logger(name: str) -> logging.Logger:
    """Retrieve a logger by name.

    Preferred over calling ``logging.getLogger()`` directly for consistent usage.
    """
    return logging.getLogger(name)


if __name__ == "__main__":
    configure_logging()
    logger = get_logger(__name__)
    logger.info("Logging test")
