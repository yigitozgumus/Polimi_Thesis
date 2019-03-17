import tensorflow as tf
from base.base_model import BaseModel


class GAN(BaseModel):

    def __init__(self, config):
        super(GAN, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        #image placeholder
        self.X = tf.placeholder(tf.float32, shape=[None]+ self.config.state_size)

        # Make the Generator model
        self.generator = tf.keras.Sequential()
        self.generator.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=self.config.noise_dim))
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU())
        self.generator.add(tf.keras.layers.Reshape(7,7,256))

        assert self.generator.output_shape == (None, 7, 7, 256)
        self.generator.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert self.generator.output_shape == (None, 7, 7, 128)
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU())

        self.generator.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert self.genertor.output_shape == (None, 14, 14, 64)
        self.generator.add(tf.keras.layers.BatchNormalization())
        self.generator.add(tf.keras.layers.LeakyReLU())

        self.generator.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert self.generor.output_shape == (None, 28, 28, 1)

        # Make the discriminator model
        self.discriminator = tf.keras.Sequential()
        self.discriminator.add(tf.keras.layers.Conv2D(64, (5,5),strides=(2, 2),padding='same'))
        self.discriminator.add(tf.keras.layers.LeakyReLU())
        self.discriminator.add(tf.keras.layers.Dropout(0.3))

        self.discriminator.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.discriminator.add(tf.keras.layers.LeakyReLU())
        self.discriminator.add(tf.keras.layers.Dropout(0.3))

        self.discriminator.add(tf.keras.layers.Flatten())
        self.discriminator.add(tf.keras.layers.Dense(1))

        # Implementation of losses
        



    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)



