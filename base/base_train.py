import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from utils.logger import Logger


class BaseTrain:
    def __init__(self, sess, model, data, config, summarizer):
        self.model = model
        self.summarizer = summarizer
        self.config = config
        log_object = Logger(self.config)
        self.logger = log_object.get_logger(__name__)
        self.sess = sess
        self.data = data
        self.init = tf.group(
            tf.global_variables_initializer(), tf.local_variables_initializer()
        )
        self.sess.run(self.init)
        self.rows = int(np.sqrt(self.config.log.num_example_imgs_to_generate))

    def train(self):
        self.logger.info("Training is started")
        for cur_epoch in range(
            self.model.cur_epoch_tensor.eval(self.sess),
            self.config.data_loader.num_epochs + 1,
            1,
        ):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self, image, cur_epoch):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
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

        plt.savefig(
            self.config.log.step_generation_dir + "image_at_epoch_{:04d}.png".format(epoch)
        )
        plt.close()
