from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_LOG_ROOT = DEFAULT_OUTPUT_ROOT / "logs"
DEFAULT_DUBBING_DIAGNOSTIC_LOG_NAME = "dubbing_diagnostics.log"
DEFAULT_VTT_GENERATION_LOG_NAME = "vtt_generation.log"

_LOGGER_LOCK = threading.Lock()


def get_log_root(log_root: Optional[Path] = None) -> Path:
    if log_root is not None:
        return Path(log_root)

    env_value = os.getenv("VIENEU_LOG_DIR")
    if env_value:
        return Path(env_value).expanduser()

    return DEFAULT_LOG_ROOT


def ensure_log_dir(log_root: Optional[Path] = None) -> Path:
    path = get_log_root(log_root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_dubbing_diagnostic_log_path(log_root: Optional[Path] = None) -> Path:
    return ensure_log_dir(log_root) / DEFAULT_DUBBING_DIAGNOSTIC_LOG_NAME


def get_vtt_generation_log_path(log_root: Optional[Path] = None) -> Path:
    return ensure_log_dir(log_root) / DEFAULT_VTT_GENERATION_LOG_NAME


def configure_file_logger(
    logger_name: str,
    *,
    log_path: Path,
    level: int = logging.INFO,
    fmt: str = "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    resolved_log_path = Path(log_path)
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    with _LOGGER_LOCK:
        for handler in logger.handlers:
            if not isinstance(handler, logging.FileHandler):
                continue
            base_filename = getattr(handler, "baseFilename", None)
            if base_filename and Path(base_filename) == resolved_log_path:
                handler.setLevel(level)
                handler.setFormatter(formatter)
                return logger

        file_handler = logging.FileHandler(resolved_log_path, encoding="utf-8", delay=True)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def configure_dubbing_diagnostic_logger(log_root: Optional[Path] = None) -> logging.Logger:
    return configure_file_logger(
        "Vieneu.Diagnostics",
        log_path=get_dubbing_diagnostic_log_path(log_root),
        level=logging.INFO,
    )
