import logging
import os


class Logger():
    def __init__(self, config,name):
        self.config = config
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(name)-15s %(levelname)-12s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=os.path.join(self.config.log.log_file_dir, "test.log"),
                            filemode='w')

        debug_handler = logging.StreamHandler()
        info_handler = logging.StreamHandler()
        debug_handler.setLevel(logging.DEBUG)
        info_handler.setLevel(logging.INFO)

        # Create formatters and add it to the handlers
        debug_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        info_format = logging.Formatter('%(name)-15s %(levelname)-12s %(message)s')
        debug_handler.setFormatter(debug_format)
        info_handler.setFormatter(info_format)

        self.logger = logging.getLogger(name)
        # Add handlers to the logger
        if self.config.log.info_logging:
            self.logger.addHandler(info_handler)
        if self.config.log.debug_logging:
            self.logger.addHandler(debug_handler)
