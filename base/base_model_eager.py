import tensorflow as tf
from utils.logger import Logger


class BaseModelEager:
    def __init__(self, config):
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()
        log_object = Logger(self.config)
        self.logger = log_object.get_logger(__name__)

    # save function that saves the checkpoint in the path defined in the config file
    def save(self):
        self.logger.info("Saving model...")
        self.checkpoint.save(self.config.log.checkpoint_prefix)
        self.logger.info("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.log.checkpoint_dir)
        if latest_checkpoint:
            self.logger.info(
                "Loading model checkpoint {} ...\n".format(latest_checkpoint)
            )
            self.checkpoint.restore(latest_checkpoint)
            self.logger.info("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        self.cur_epoch_tensor = tf.Variable(0, name="cur_epoch")

    def increment_cur_epoch_tensor(self):
        tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        self.global_step_tensor = tf.Variable(0, name="global_step")

    def init_saver(self):
        # just copy the following line in your child class
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
