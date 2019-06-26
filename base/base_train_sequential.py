import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.logger import Logger


class BaseTrainSequential:
    def __init__(self, sess, model, data, config, summarizer):
        self.model = model
        self.summarizer = summarizer
        self.config = config
        log_object = Logger(self.config)
        self.logger = log_object.get_logger(__name__)
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)
        self.rows = int(np.sqrt(self.config.log.num_example_imgs_to_generate))

    def train(self):
        self.logger.info("Training of WGAN is started")
        for cur_epoch in range(
            self.model.cur_epoch_tensor.eval(self.sess),
            self.config.data_loader.num_epochs_gan + 1,
            1,
        ):
            self.train_epoch_gan()
            self.sess.run(self.model.increment_cur_epoch_tensor)
        if self.config.trainer.reset_first_counter:
            self.sess.run(self.model.reset_cur_epoch_tensor)

        self.logger.info("Training of Generator Encoder is started")
        for cur_epoch in range(
            self.model.cur_epoch_tensor.eval(self.sess),
            self.config.data_loader.num_epochs_enc_gen + 1,
            1,
        ):
            self.train_epoch_enc_gen()
            self.sess.run(self.model.increment_cur_epoch_tensor)

        self.sess.run(self.model.reset_cur_epoch_tensor)

        self.logger.info("Training of Reconstructer Encoder is started")
        for cur_epoch in range(
            self.model.cur_epoch_tensor.eval(self.sess),
            self.config.data_loader.num_epochs_enc_rec + 1,
            1,
        ):
            self.train_epoch_enc_rec()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch_gan(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_epoch_enc_gen(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_epoch_enc_rec(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step_gan(self, image, cur_epoch):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

    def train_step_enc_gen(self, image, cur_epoch):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

    def train_step_enc_rec(self, image, cur_epoch):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

    def test(self):
        self.logger.info("Testing is started")
        self.test_epoch()

    def test_epoch(self):
        raise NotImplementedError

    def save_generated_images(self, predictions, epoch):
        # make sure the training parameter is set to False because we
        # don't want to train the batchnorm layer when doing inference.
        predictions = np.asarray(predictions)[0]
        fig = plt.figure(figsize=(self.rows, self.rows))
        for i in range(predictions.shape[0]):
            plt.subplot(self.rows, self.rows, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
            plt.axis("off")

        plt.savefig(self.config.log.step_generation_dir + "image_at_epoch_{:04d}.png".format(epoch))
        p
