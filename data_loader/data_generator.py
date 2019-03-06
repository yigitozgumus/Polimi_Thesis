import numpy as np
from utils.DataLoader import DataLoader
import tensorflow as tf

class DataGenerator():
    def __init__(self, config):
        self.config = config
        # load data here
        d = DataLoader(self.config.data_folder)
        # Get the filenames and labels
        self.filenames, self.labels = d.get_sub_dataset(self.config.image_size)
        # Create the Dataset using Tensorflow Data API
        self.dataset = tf.data.Dataset.from_tensor_slices((self.filenames, self.labels))
        # Apply parse function to get the numpy array of the images
        self.dataset = self.dataset.map(self._parse_function)
        # Apply batching
        self.dataset.batch(config.batch_size)
        # Create the iterator
        self.iterator = self.dataset.make_initializable_iterator()
        # Create the next element
        self.next_element,self.next_label = self.iterator.get_next()


    def _parse_function(self,filename, label):
        # Read the image
        image_string = tf.read_file(filename)
        # Decode the image
        image_decoded = tf.image.decode_jpeg(image_string)
        # Resize the image --> 28 is default
        #TODO
        image_resized = tf.image.resize_images(image_decoded, [28, 28])
        # Normalize the values of the pixels
        image_normalized = tf.image.convert_image_dtype(image_resized,dtype=float,name="scaling")
        # Random image flip left-right
        image_random_flip_lr = tf.image.random_flip_left_right(image_normalized,seed=tf.random.set_random_seed(1234))
        # Random image flip up-down
        image_random_flip_ud = tf.image.random_flip_up_down(image_random_flip_lr,seed=tf.random.set_random_seed(1234))
        return image_random_flip_ud, label

