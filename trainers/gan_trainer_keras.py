from base.base_train_keras import BaseTrain_keras
from tqdm import tqdm
import numpy as np


class GANTrainer_keras(BaseTrain_keras):
    def __init__(self, sess, model, data, config, logger):
        super(GANTrainer_keras, self).__init__(
            sess, model, data, config, logger)

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop on the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        pass


