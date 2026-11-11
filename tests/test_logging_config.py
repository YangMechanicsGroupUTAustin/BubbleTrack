"""Tests for bubbletrack.logging_config."""

from __future__ import annotations

import logging

import pytest

from bubbletrack.logging_config import setup_logging


@pytest.fixture(autouse=True)
def _clean_handlers():
    """Remove all handlers from the bubbletrack logger after each test."""
    yield
    logger = logging.getLogger("bubbletrack")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)  # reset to default


def test_setup_creates_logger():
    """setup_logging creates a console handler at INFO level."""
    setup_logging()
    logger = logging.getLogger("bubbletrack")

    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_file_handler_created(tmp_path):
    """setup_logging with log_dir creates both console and file handlers."""
    setup_logging(log_dir=tmp_path / "logs")

    logger = logging.getLogger("bubbletrack")
    assert len(logger.handlers) == 2

    handler_types = {type(h) for h in logger.handlers}
    assert logging.StreamHandler in handler_types
    assert logging.FileHandler in handler_types

    # Verify log directory was created
    assert (tmp_path / "logs").is_dir()

    # Verify log file was created
    assert (tmp_path / "logs" / "bubbletrack.log").exists()


def test_custom_level():
    """setup_logging respects a custom log level."""
    setup_logging(level=logging.DEBUG)
    logger = logging.getLogger("bubbletrack")

    assert logger.level == logging.DEBUG


def test_setup_twice_does_not_duplicate_handlers(tmp_path):
    """Repeated setup (e.g. the CLI called twice in one process) must not
    print every log line several times."""
    setup_logging(log_dir=tmp_path / "logs")
    setup_logging(log_dir=tmp_path / "logs")
    logger = logging.getLogger("bubbletrack")
    assert len(logger.handlers) == 2
