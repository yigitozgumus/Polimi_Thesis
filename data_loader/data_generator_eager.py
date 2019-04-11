from utils.DataLoader import DataLoader
import tensorflow as tf


class DataGeneratorEager:
    def __init__(self, config):
        """
        Args:
            config: config file of the current model
        """

        self.config = config
        # load data here
        d = DataLoader(self.config)
        # Get the filenames and labels
        self.filenames = d.get_train_dataset()
        # Create the Dataset using Tensorflow Data API
        self.dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        # Apply parse function to get the numpy array of the images
        self.dataset = self.dataset.map(
            map_func=self._parse_function,
            num_parallel_calls=self.config.data_loader.num_parallel_calls,
        )
        # Shuffle the dataset
        # self.dataset = self.dataset.shuffle(self.config.buffer_size)
        # Repeat the dataset indefinitely
        self.dataset = self.dataset.repeat(self.config.data_loader.num_epochs)
        # Applying prefetch to increase the performance
        # Prefetch the next 10 batches
        self.dataset = self.dataset.prefetch(
            buffer_size=10 * config.data_loader.batch_size
        )
        # Apply batching
        self.dataset = self.dataset.batch(config.data_loader.batch_size)

    def _parse_function(self, filename):
        # Read the image
        """
        Args:
            filename: image file to be parsed
        """

        # Read the image file
        image_file = tf.io.read_file(filename)
        # Decode the image
        image_decoded = tf.io.decode_jpeg(image_file)
        # Resize the image --> 28 is default
        image_resized = tf.image.resize(
            image_decoded,
            [self.config.data_loader.image_size, self.config.data_loader.image_size],
        )
        # Normalize the values of the pixels. The function that is applied is below
        # (x - mean) / adjusted_stddev
        # adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))
        image_normalized = tf.image.per_image_standardization(image_resized)
        # Random image flip left-right
        image_random_flip_lr = tf.image.random_flip_left_right(
            image_normalized, seed=1234
        )
        # Random image flip up-down
        image_random_flip_ud = tf.image.random_flip_up_down(
            image_random_flip_lr, seed=1234
        )

        return image_random_flip_ud
