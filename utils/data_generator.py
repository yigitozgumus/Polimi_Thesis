import numpy as np
from utils.DataLoader import DataLoader
import tensorflow as tf


class DataGenerator():
    def __init__(self, config):
        """
        Args:
            config: config file of the current model
        """

        self.config = config
        # load data here
        d = DataLoader(self.config.data_folder)
        # Get the filenames and labels
        self.filenames, self.labels = d.get_sub_dataset(self.config.image_size)
       # assert len(self.filenames) == len(self.labels)
        # Create the Dataset using Tensorflow Data API
        self.dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        # Apply parse function to get the numpy array of the images
        self.dataset = self.dataset.map(map_func=self._parse_function,
                                        num_parallel_calls=self.config.num_parallel_calls)
        # Shuffle the dataset
        self.dataset = self.dataset.shuffle(self.config.buffer_size)
        # Repeat the dataset indefinitely
        self.dataset = self.dataset.repeat(self.config.num_epochs)
        # Apply batching
        self.dataset = self.dataset.batch(config.batch_size)
        # Applying prefetch to increase the performance
        # Prefetch the next 10 batches
        self.dataset = self.dataset.prefetch(
            buffer_size=10 * config.batch_size)

    def _parse_function(self, filename):
        # Read the image
        """
        Args:
            filename: image file to be parsed
        """

        # Read the image file
        image_file = tf.read_file(filename)
        # Decode the image
        image_decoded = tf.image.decode_image(image_file, channels=1)
        # Resize the image --> 28 is default
        # TODO
        image_resized = tf.image.resize_images(image_decoded, [28, 28])
        # Normalize the values of the pixels
        image_normalized = tf.image.convert_image_dtype(
            image_resized, dtype=float, name="scaling")
        # Random image flip left-right
        image_random_flip_lr = tf.image.random_flip_left_right(
            image_normalized, seed=tf.random.set_random_seed(1234))
        # Random image flip up-down
        image_random_flip_ud = tf.image.random_flip_up_down(
            image_random_flip_lr, seed=tf.random.set_random_seed(1234))
        return image_random_flip_ud
