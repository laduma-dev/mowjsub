import logging
from importlib import metadata
from types import SimpleNamespace

__version__ = metadata.version(__package__)

#: Command names. These used to be an OmegaConf object, which was a whole
#: config system for three constants.
BIN = SimpleNamespace(
    im_plane="im-mowjsub",
    vis_plane="vis-mowjsub",
    doppler_plane="doppler-mowjsub",
)

#: Every module logs to this one logger. It used to be one per entry point,
#: which meant `utils` -- shared by all three -- logged to the image-plane
#: logger whichever command you had actually run.
LOGGER = "mowjsub"

#: What `--loglevel` accepts. 'trace' is not a Python level; it is kept because
#: the option has always offered it, and maps to DEBUG.
LOG_LEVELS = {
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "trace": logging.DEBUG,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def set_logger(level="info", name=LOGGER):
    """Configure and return the package logger.

    Replaces ``scabha.init_logger``. Library modules take their logger with
    ``logging.getLogger`` and never attach a handler; an entry point calls this
    once, so importing mowjsub configures nothing on its own.

    Args:
        level (str|int): Level name from :data:`LOG_LEVELS`, or a logging level.
        name (str): Logger to configure.

    Returns:
        logging.Logger
    """
    if isinstance(level, str):
        level = LOG_LEVELS.get(level.lower(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    for handler in logger.handlers:
        handler.setLevel(level)

    return logger
