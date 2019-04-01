import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class BaseTrainEager:

    def __init__(self, model,data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.data = data
        self.rows = int(np.sqrt(self.config.num_example_imgs_to_generate))

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.numpy(), self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.model.increment_cur_epoch_tensor()

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self,image, cur_epoch):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

    def save_generated_images(self,predictions, epoch):
        # make sure the training parameter is set to False because we
        # don't want to train the batchnorm layer when doing inference.
        predictions = np.asarray(predictions)
        fig = plt.figure(figsize=(self.rows, self.rows))
        for i in range(predictions.shape[0]):
            plt.subplot(self.rows, self.rows, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(self.config.step_generation_dir + 'image_at_epoch_{:04d}.png'.format(epoch))

