import numpy as np
from utils.DataLoader import DataLoader


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        d = DataLoader(self.config.data_folder)
        self.input_data, self.labels = d.get_sub_dataset(self.config.image_size)

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]
