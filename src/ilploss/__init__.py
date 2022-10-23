import logging

from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler(omit_repeated_times=False, show_level=False))
logger.setLevel(logging.DEBUG)
logger.propagate = False
