"""
Contains functionality for setting up logging.

Logger name selection:
- If no name is provided, derive it from the calling script/module filename
  so log lines show the actual script
"""
import logging
from logging.handlers import RotatingFileHandler
import inspect
import os
import sys

def setup_logger(log_filename: str = "app.log", name: str | None = None):
    """
    Sets up a logger that logs to both console and a file.

    Args:
        log_filename: Path to the file to write logs to.
        name: Optional logger name. If None, uses the caller's module filename
              (e.g., "train_model") so %(name)s in log lines reflects the script.
    """
    if name is None:
        # Derive name from the caller's module/script
        frame = inspect.stack()[1].frame
        module = inspect.getmodule(frame)
        if module and getattr(module, "__file__", None):
            name = os.path.splitext(os.path.basename(module.__file__))[0]
        else:
            name = os.path.splitext(os.path.basename(sys.argv[0] or "app"))[0]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    file_handler = RotatingFileHandler(log_filename, maxBytes=5_000_000, backupCount=3)

    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    # Include date and hour:min:sec in all log lines
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # expose shorthand functions bound to this logger
    global debug, info, warning, error, critical
    debug = logger.debug
    info = logger.info
    warning = logger.warning
    error = logger.error
    critical = logger.critical

    return logger
