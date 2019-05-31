import tensorflow as tf
import os


class Summarizer:
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.train_summary_writer = tf.summary.FileWriter(
            os.path.join(self.config.log.summary_dir, "train"), self.sess.graph
        )
        self.valid_summary_writer = tf.summary.FileWriter(
            os.path.join(self.config.log.summary_dir, "valid"), self.sess.graph
        )
        self.valid_summary_writer_2 = tf.summary.FileWriter(
            os.path.join(self.config.log.summary_dir, "valid_2"), self.sess.graph
        )

    # it can summarize scalars and images.
    def add_tensorboard(self, step, summarizer="train", scope="", summaries=None):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the test one
        :param scope: variable scope
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """
        if summarizer == "train":
            summary_writer = self.train_summary_writer
        elif summarizer == "valid":
            summary_writer = self.valid_summary_writer
        elif summarizer == "valid_2":
            summary_writer = self.valid_summary_writer_2
        with tf.variable_scope(scope):
            for summary in summaries:
                summary_writer.add_summary(summary, step)
            summary_writer.flush()
