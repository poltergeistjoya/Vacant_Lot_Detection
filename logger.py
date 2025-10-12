import logging
import sys

def get_logger(name:str = "vacant_lots") -> logging.Logger:
    """
    Returns shared logger config to print to stdout
    If alr configured, reuses same logger instance
    Safe for ipynb notebooks
    """
    logger = logging.getLogger(name) #one logger throughout system
    logger.setLevel(logging.INFO) 


    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
