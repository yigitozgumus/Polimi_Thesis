from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from time import sleep
from time import time
from utils.evaluations import do_prc


class SkipGANomalyTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, summarizer):
        super(SkipGANomalyTrainer, self).__init__(sess, model, data, config, summarizer)
        self.batch_size = self.config.data_loader.batch_size
        self.noise_dim = self.config.trainer.noise_dim
        self.img_dims = self.config.trainer.image_dims
        # Inititalize the train Dataset Iterator
        self.sess.run(self.data.iterator.initializer)
        # Initialize the test Dataset Iterator
        self.sess.run(self.data.test_iterator.initializer)

    def train_epoch(self):
        pass

    def train_step(self, image, cur_epoch):
        pass
