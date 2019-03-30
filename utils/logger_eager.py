import tensorflow as tf
import os


class Logger_eager:
    def __init__(self,config):
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.train_summary_writer = tf.contrib.summary.create_file_writer(os.path.join(self.config.summary_dir, "train"))
        with self.train_summary_writer.as_default():
            tf.contrib.summary.always_record_summaries()
        self.test_summary_writer = tf.contrib.summary.create_file_writer(os.path.join(self.config.summary_dir, "test"))

    # it can summarize scalars and images.
    def summarize(self, step, summarizer="train", scope="", summaries_dict=None):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the test one
        :param scope: variable scope
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """
        summary_writer = self.train_summary_writer if summarizer == "train" else self.test_summary_writer
        with tf.variable_scope(scope):
            if summaries_dict is not None:
                with tf.contrib.summary.record_summaries_every_n_global_steps(1):
                    for tag, value in summaries_dict.items():
                        if len(value.shape) <= 1:
                            tf.contrib.summary.scalar(tag, value)
                        else:
                            tf.contrib.summary.image(tag, value)

