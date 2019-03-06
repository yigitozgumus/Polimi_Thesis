import tensorflow as tf

tf.enable_eager_execution()

from base.base_model import BaseModel


class Gan(BaseModel):

    def __init__(self, config):
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def build_model(self):
        pass
