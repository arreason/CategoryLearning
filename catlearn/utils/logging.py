"""Module with generic tools"""
import os
import warnings
import logging
import torch


def init_logger():
    """Initialize events logger."""
    LOGGING_FILE = 'data_utils_debug.log'
    if os.path.exists(os.path.join(os.getcwd(), LOGGING_FILE)):
        os.remove(os.path.join(os.getcwd(), LOGGING_FILE))
    logging.basicConfig(format='%(asctime)s %(message)s',
        filename=LOGGING_FILE,
        level=logging.DEBUG
    )
    logging.debug('\nStart logging.')

#pylint: disable=unused-variable
def str_color(color_type: str, string: str) -> str:
    """Function to format strings with color and/or bold font."""
    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
    COLORS = {
        'WARNING': YELLOW,
        'INFO': WHITE,
        'DEBUG': BLUE,
        'CRITICAL': YELLOW,
        'ERROR': RED,
    }
    if color_type == 'W':
        return COLOR_SEQ % (30 + COLORS['WARNING']) + string + RESET_SEQ
    warnings.warn('Unknown color is specified in str_color(). Raw string returned.', UserWarning)
    return string
