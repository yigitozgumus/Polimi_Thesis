import numpy as np
from utils.DataLoader import DataLoader
import tensorflow as tf
from utils.logger import Logger


class DataGenerator:
    def __init__(self, config):
        """
        Args:
            config: config file of the current model
        """

        self.config = config
        log_object = Logger(self.config)
        self.logger = log_object.get_logger(__name__)
        # load data here
        d = DataLoader(self.config)
        self.logger.info("Data is loading...")
        # Get the filenames and labels
        self.filenames_train = d.get_train_dataset()
        # assert len(self.filenames) == len(self.labels)
        # Create the Dataset using Tensorflow Data API
        self.dataset = tf.data.Dataset.from_tensor_slices(self.filenames_train)
        # Apply parse function to get the numpy array of the images
        self.dataset = self.dataset.map(
            map_func=self._parse_function,
            num_parallel_calls=self.config.data_loader.num_parallel_calls,
        )
        # Shuffle the dataset
        if self.config.data_loader_validation:
            buffer_size = int(
                self.config.data_loader.buffer_size
                * ((100 - self.config.data_loader.validation_percent) / 100)
            )
        else:
            buffer_size = self.config.data_loader.buffer_size
        self.dataset = self.dataset.shuffle(buffer_size)
        # Repeat the dataset indefinitely
        self.dataset = self.dataset.repeat()
        # Apply batching
        self.dataset = self.dataset.batch(self.config.data_loader.batch_size)
        # Applying prefetch to increase the performance
        # Prefetch the next 10 batches
        self.dataset = self.dataset.prefetch(buffer_size=10 * self.config.data_loader.batch_size)
        self.iterator = self.dataset.make_initializable_iterator()
        self.image = self.iterator.get_next()

        # Validation Dataset
        if self.config.data_loader.validation:
            self.filenames_valid = d.get_valid_dataset()
            # Create the Dataset using Tensorflow Data API
            self.valid_dataset = tf.data.Dataset.from_tensor_slices(self.filenames_valid)
            # Apply parse function to get the numpy array of the images
            self.valid_dataset = self.valid_dataset.map(
                map_func=self._parse_function,
                num_parallel_calls=self.config.data_loader.num_parallel_calls,
            )
            buffer_size = int(
                (self.config.data_loader.buffer_size * self.config.data_loader.validation_percent)
                / 100
            )
            self.valid_dataset = self.valid_dataset.shuffle(buffer_size)
            self.valid_dataset = self.valid_dataset.repeat()
            # Apply batching
            self.valid_dataset = self.valid_dataset.batch(buffer_size)
            self.valid_iterator = self.valid_dataset.make_initializable_iterator()
            self.valid_image = self.valid_iterator.get_next()

        # If the mode is anomaly create the test dataset
        if self.config.data_loader.mode == "anomaly":
            self.test_filenames, self.test_labels = d.get_test_dataset()
            self.test_dataset = tf.data.Dataset.from_tensor_slices(
                (self.test_filenames, self.test_labels)
            )
            self.test_dataset = self.test_dataset.map(
                map_func=self._parse_function_test,
                num_parallel_calls=self.config.data_loader.num_parallel_calls,
            )
            # Shuffle the dataset
            # self.test_dataset = self.test_dataset.shuffle(self.config.data_loader.buffer_size)
            # Repeat the dataset indefinitely
            self.test_dataset = self.test_dataset.repeat()
            # Apply batching
            self.test_dataset = self.test_dataset.batch(self.config.data_loader.test_batch)
            self.test_iterator = self.test_dataset.make_initializable_iterator()
            self.test_image, self.test_label = self.test_iterator.get_next()
        if self.config.data_loader.mode == "visualization":
            self.test_filenames, self.test_labels, self.ground_truth = d.get_test_dataset_vis()
            self.test_dataset = tf.data.Dataset.from_tensor_slices(
                (self.test_filenames, self.test_labels, self.ground_truth)
            )
            self.test_dataset = self.test_dataset.map(
                map_func=self._parse_function_test_2,
                num_parallel_calls=self.config.data_loader.num_parallel_calls,
            )
            # Shuffle the dataset
            # self.test_dataset = self.test_dataset.shuffle(self.config.data_loader.buffer_size)
            # Repeat the dataset indefinitely
            self.test_dataset = self.test_dataset.repeat()
            # Apply batching
            self.test_dataset = self.test_dataset.batch(self.config.data_loader.test_batch)
            self.test_iterator = self.test_dataset.make_initializable_iterator()
            self.test_image, self.test_label, self.ground_truth = self.test_iterator.get_next()

    def _parse_function(self, filename):
        # Read the image
        """
        Args:
            filename: image file to be parsed
        """

        # Read the image file
        image_file = tf.read_file(filename)
        # Decode the image
        image_decoded = tf.image.decode_jpeg(image_file)
        # Resize the image --> 28 is default
        image_resized = tf.image.resize_images(
            image_decoded, [self.config.data_loader.image_size, self.config.data_loader.image_size]
        )
        # Normalize the values of the pixels. The function that is applied is below
        # (x - mean) / adjusted_stddev
        # adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))
        image_normalized = tf.image.per_image_standardization(image_resized)
        #image_normalized = image_resized / 255.0
        # Random image flip left-right
        image_random_flip_lr = tf.image.random_flip_left_right(
            image_normalized, seed=tf.random.set_random_seed(self.config.data_loader.random_seed)
        )
        # Random image flip up-down
        image_random_flip_ud = tf.image.random_flip_up_down(
            image_random_flip_lr,
            seed=tf.random.set_random_seed(self.config.data_loader.random_seed),
        )
        return image_random_flip_ud

    def _parse_function_test(self, img_file, tag):
        # Read the image
        img = tf.read_file(img_file)
        # Decode the image and the label
        img_decoded = tf.image.decode_jpeg(img)
        image_resized = tf.image.resize_images(
            img_decoded, [self.config.data_loader.image_size, self.config.data_loader.image_size]
        )
        image_normalized = tf.image.per_image_standardization(image_resized)
        #image_normalized = image_resized / 255.0

        return image_normalized, tag
    
    def _parse_function_test_2(self, img_file, tag, ground):
        # Read the image
        img = tf.read_file(img_file)
        ground = tf.read_file(ground)
        # Decode the image and the label
        img_decoded = tf.image.decode_jpeg(img)
        ground_decoded = tf.image.decode_jpeg(ground)
        image_resized = tf.image.resize_images(
            img_decoded, [self.config.data_loader.image_size, self.config.data_loader.image_size]
        )
        ground_resized = tf.image.resize_images(
            ground_decoded, [self.config.data_loader.image_size, self.config.data_loader.image_size]
        )
        image_normalized = image_resized / 255.0
        ground_normalized = tf.image.per_image_standardization(ground_resized)
        

        return image_normalized, tag, ground_normalized
