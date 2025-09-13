"""
Contains functionality for setting up logging
"""
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(log_filename="app.log"):
    """
    Sets up a logger that logs to both console and a file.
    """
    logger = logging.getLogger("my_app")
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    file_handler = RotatingFileHandler(log_filename, maxBytes=5_000_000, backupCount=3)

    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
