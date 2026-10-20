import logging
import config
from pathlib import Path
from enum import Enum

class LODNMLogger:
    
    class ColoredFormatter(logging.Formatter):
        class EColor(Enum):
            RED = '\033[31m'
            BOLD_RED = '\033[1;31m'
            WHITE = '\033[0m'
            GREEN = '\033[32m'
            BLUE = '\033[34m'
            YELLOW = '\033[33m'
        COLORS = {
            logging.INFO: EColor.GREEN.value,
            logging.DEBUG: EColor.BLUE.value,
            logging.WARNING: EColor.YELLOW.value,
            logging.ERROR: EColor.RED.value,
            logging.CRITICAL: EColor.BOLD_RED.value,
        }

        def format(self, record):
            color = self.COLORS.get(record.levelno, self.EColor.WHITE.value)
            message = super().format(record)
            return f'{color}{message}{self.EColor.WHITE.value}'

    class DestinationFilter(logging.Filter):
        def __init__(self, destination) -> None:
            super().__init__()
            self.destination = destination

        def filter(self, record) -> bool:
            return (record.destination == self.destination 
                    or record.destination == 'both')


    def __init__(self, name_logger: str, name_file_log: str='', ext_log: str='.log', dir_file_log: str='', logging_format: str='') -> None:
        self.logger = logging.getLogger(name_logger)
        self.logger.setLevel(logging.DEBUG)
        #config
        self.path_log = Path(dir_file_log if len(dir_file_log) else '.\\' + config.LOGS_DIR)
        self.path_log.mkdir(parents=True, exist_ok=True)
        self.name_file_log = (name_file_log if len(name_file_log) else f'log{sum(1 for item in self.path_log.iterdir() if item.is_dir())}')
        self.path_log = self.path_log / (self.name_file_log + ext_log)
        self.logging_format = logging_format if len(logging_format) else config.LOGGING_FORMAT
        #Console handler
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.DEBUG)
        self.console_handler.addFilter(self.DestinationFilter("console"))
        self.console_handler.setFormatter(self.ColoredFormatter(config.LOGGING_FORMAT))
        #File handler
        self.file_handler = logging.FileHandler(str(self.path_log))
        self.file_handler.addFilter(self.DestinationFilter("file"))
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(logging.Formatter(self.logging_format))
        
        #add handlers
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.file_handler)

    def debug(self, message, *, destination='console', **kwargs):
        self.logger.debug(message,
                            extra={'destination': destination},
                            **kwargs)
    def info(self, message, *, destination='console', **kwargs):
            self.logger.info(message,
                                extra={'destination': destination},
                                **kwargs)
    def warning(self, message, *, destination='console', **kwargs):
            self.logger.warning(message,
                                extra={'destination': destination},
                                **kwargs)
    def error(self, message, *, destination='console', **kwargs):
                self.logger.error(message,
                                    extra={'destination': destination},
                                    **kwargs)
    def critical(self, message, *, destination='console', **kwargs):
                self.logger.critical(message,
                                    extra={'destination': destination},
                                    **kwargs)