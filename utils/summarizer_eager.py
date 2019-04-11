import tensorflow as tf
import os


class Summarizer_eager:
    def __init__(self, config):
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.train_summary_writer = tf.summary.create_file_writer(
            os.path.join(self.config.summary_dir, "train")
        )
        self.test_summary_writer = tf.summary.create_file_writer(
            os.path.join(self.config.summary_dir, "test")
        )
