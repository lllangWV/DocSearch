import os
from datetime import datetime
import logging


LOG_DIR = os.path.join('C:/Users/lllang/Desktop/Current_Projects/WVU_Accreditation_Requests/logs')
class InfoFilter(logging.Filter):
    def filter(self, record):
        # Only allow INFO level messages
        return record.levelno == logging.INFO

PARENT_LOGGER_NAME = 'matgraphdb'

FORMAT_STRING = '%(name)s :: %(levelname)-8s :: %(message)s'
FORMATTER = logging.Formatter(fmt=FORMAT_STRING)

logging_levels=['critical','error','warning','info','debug']

def setup_logging(log_dir=LOG_DIR, console_out=True, log_level='info', format=FORMATTER):
    """
    Set up logging for the MatGraphDB package.

    Args:
        log_dir (str): The directory where the log file will be saved.

    Returns:
        logger (logging.Logger): The logger object for the MatGraphDB package.
    """

    # Create a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"package_{timestamp}.log"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filepath = os.path.join(log_dir,f'{PARENT_LOGGER_NAME}.log')


    logging.basicConfig(level=logging.INFO, 
                        format=FORMAT_STRING)

    logger = logging.getLogger(PARENT_LOGGER_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)
    
    handler = logging.FileHandler(os.path.join(LOG_DIR, f'{PARENT_LOGGER_NAME}.log'))
    logger.addHandler(handler)

    return logger


def get_parent_logger(console_out=False, log_level='info', format=FORMATTER):
    if log_level not in logging_levels:
        raise ValueError(f"Invalid log level: {log_level}. Valid log levels are: {logging_levels}")
    
    if log_level == 'debug':
        log_level = logging.DEBUG
    elif log_level == 'info':
        log_level = logging.INFO
    elif log_level == 'warning':
        log_level = logging.WARNING
    elif log_level == 'error':
        log_level = logging.ERROR
    elif log_level == 'critical':
        log_level = logging.CRITICAL

    logger = logging.getLogger(PARENT_LOGGER_NAME)

    if console_out:
        handler = logging.StreamHandler() 
    else:
        handler = logging.FileHandler(os.path.join(LOG_DIR, f'{PARENT_LOGGER_NAME}.log'))
        handler.setFormatter(format)

    logger.setHandler(handler)
    logger.setLevel(log_level)
    

    return logger

def get_child_logger(name, console_out=False, log_level='info', format=FORMATTER):
    if log_level not in logging_levels:
        raise ValueError(f"Invalid log level: {log_level}. Valid log levels are: {logging_levels}")
    
    if log_level == 'debug':
        log_level = logging.DEBUG
    elif log_level == 'info':
        log_level = logging.INFO
    elif log_level == 'warning':
        log_level = logging.WARNING
    elif log_level == 'error':
        log_level = logging.ERROR
    elif log_level == 'critical':
        log_level = logging.CRITICAL

    child_logger = logging.getLogger(f'{PARENT_LOGGER_NAME}.{name}')
    child_logger.propagate = False  # Prevent propagation to parent logger

    if console_out:
        child_handler = logging.StreamHandler() 
        child_handler.setLevel(log_level)
        child_logger.addHandler(child_handler)
    else:
        
        if log_level == 'debug':
            child_error_handler = logging.FileHandler(os.path.join(LOG_DIR, f'{name}_error.log'))
            child_error_handler.setLevel(logging.DEBUG)
            child_error_handler.setFormatter(format)

            child_logger.addHandler(child_error_handler)

        child_handler = logging.FileHandler(os.path.join(LOG_DIR, f'{name}.log'))
        child_handler.setLevel(log_level)
        child_handler.setFormatter(format)

        child_logger.addHandler(child_handler)

    return child_logger

setup_logging(log_dir=LOG_DIR, console_out=True, log_level='info')
logger=get_child_logger('wvu_query',console_out=False, log_level='info')


logger.info('info')

logger.debug('debug')

logger.error('error')