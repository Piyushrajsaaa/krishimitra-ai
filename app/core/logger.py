"""
KrishiMitra AI — Centralized Logger
Writes structured logs to both console and file.
Supports audit logging for Responsible AI compliance.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from app.core.config import settings


def setup_logger(name: str = "krishimitra") -> logging.Logger:
    """
    Creates and configures a logger with:
    - Console handler (colored, human-readable)
    - File handler (structured, persistent)
    """

    # Ensure logs directory exists
    log_dir = Path(settings.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # ── Formatter ─────────────────────────────────────────
    fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    # ── Console Handler ───────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # ── File Handler ──────────────────────────────────────
    file_handler = logging.FileHandler(settings.LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_audit_logger() -> logging.Logger:
    """
    Separate audit logger for Responsible AI compliance.
    Logs every prediction request (no PII stored).
    """
    audit_log_path = Path("logs/audit.log")
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    audit_logger = logging.getLogger("krishimitra.audit")
    audit_logger.setLevel(logging.INFO)

    if audit_logger.handlers:
        return audit_logger

    fmt = "[%(asctime)s] [AUDIT] %(message)s"
    formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(audit_log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    audit_logger.addHandler(file_handler)

    return audit_logger


# ── Global logger instances ────────────────────────────────
logger = setup_logger("krishimitra")
audit_logger = get_audit_logger()
