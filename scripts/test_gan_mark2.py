 
import tensorflow as tf

tf.enable_eager_execution()

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time

from IPython import display

import numpy as np
from utils.DataLoader import DataLoader
import tensorflow as tf
BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

class DataGenerator():
    def __init__(self):
       # self.config = config
        # load data here
        d = DataLoader("./data")
        # Get the filenames and labels
        self.filenames, self.labels = d.get_sub_dataset(28)
        #assert len(self.filenames) == len(self.labels)
        # Create the Dataset using Tensorflow Data API
        self.dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        # Apply parse function to get the numpy array of the images
        self.dataset = self.dataset.map(self._parse_function)
        # Shuffle the dataset
        self.dataset = self.dataset.shuffle(self.filenames.get_shape()[0])
        # Apply batching
        self.dataset = self.dataset.batch(50)
        # Create the iterator
        #self.iterator = self.dataset.make_initializable_iterator()
        # Create the next element
      #  self.next_element, self.next_label = self.iterator.get_next()

    def _parse_function(self, filename):
        # Read the image
        image_string = tf.io.read_file(filename)
        # Decode the image
        image_decoded = tf.image.decode_jpeg(image_string)
        # Resize the image --> 28 is default
        # TODO
        image_resized = tf.image.resize(image_decoded, [28, 28])
        # Normalize the values of the pixels
        image_normalized = tf.image.convert_image_dtype(image_resized, dtype=float, name="scaling")
        # Random image flip left-right
        image_random_flip_lr = tf.image.random_flip_left_right(image_normalized, seed=tf.random.set_seed(1234))
        # Random image flip up-down
        image_random_flip_ud = tf.image.random_flip_up_down(image_random_flip_lr, seed=tf.random.set_seed(1234))
        return image_random_flip_ud



def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model

def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)


def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss

@tf.contrib.eager.defun
def train_step(images):
    # generating noise from a normal distribution
    noise = tf.random_normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        generated_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(generated_output)
        disc_loss = discriminator_loss(real_output, generated_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,discriminator.variables))

def generate_and_save_images(model, epoch, test_input):
    # make sure the training parameter is set to False because we
    # don't want to train the batchnorm layer when doing inference.
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()



def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for images in dataset:
            train_step(images)

        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                epoch + 1,
                                random_vector_for_generation)

        # saving (checkpoint) the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                        time.time()-start))
  # generating after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                           epochs,
                           random_vector_for_generation)




dataset = DataGenerator()
generator = make_generator_model()
discriminator = make_discriminator_model()
generator_optimizer = tf.train.AdamOptimizer(1e-4)
discriminator_optimizer = tf.train.AdamOptimizer(1e-4)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


    # We'll re-use this random vector used to seed the generator so
    # it will be easier to see the improvement over time.
random_vector_for_generation = tf.random_normal([num_examples_to_generate,
                                                     noise_dim])

    #train_step = tf.contrib.eager.defun(train_step)
train(dataset.dataset, EPOCHS)





