
import logging

class Logger():
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.config.log.name)

    def initialize(self):
        # Create the handlers
        debug_handler = logging.StreamHandler()
        info_handler = logging.StreamHandler()
        f_handler = logging.FileHandler("test.log")
        debug_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.ERROR)
        info_handler.setLevel(logging.INFO)

        # Create formatters and add it to the handlers
        debug_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        info_format = logging.Formatter('%(levelname)s - %(message)s')
        debug_handler.setFormatter(debug_format)
        f_handler.setFormatter(f_format)
        info_handler.setFormatter(info_format)

        # Add handlers to the logger
        if self.config.log.file_logging:
            self.logger.addHandler(f_handler)
        if self.config.log.info_logging:
            self.logger.addHandler(info_handler)
        if self.config.log.debug_logging:
            self.logger.addHandler(debug_handler)

        return self.logger
