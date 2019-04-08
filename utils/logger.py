import logging
import os
import sys

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
LOG_FILE = "experiment_run.log"


class Logger:
    def __init__(self, config):
        self.config = config
        self.log_file = os.path.join(self.config.log.log_file_dir, LOG_FILE)

    def get_logger(self, logger_name):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)  # better to have too much log than not enough
        logger.addHandler(self.get_console_handler())
        logger.addHandler(self.get_file_handler())
        # with this pattern, it's rarely necessary to propagate the error up to parent
        logger.propagate = False
        return logger

    def get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(FORMATTER)
        return console_handler

    def get_file_handler(self):
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(FORMATTER)
        file_handler.setLevel(logging.INFO)
        return file_handler
