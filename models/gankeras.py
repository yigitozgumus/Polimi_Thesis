import tensorflow as tf

from base.base_model_keras import BaseModelKeras
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation, Flatten, Dropout
from tensorflow.keras.models import Model


class GanKeras(BaseModelKeras):

    def __init__(self, config):
        super(GanKeras, self).__init__(config)
        self.build_model()


    def build_model(self):

        self.optimizer = tf.train.AdamOptimizer(self.config.discriminator_l_rate,
                                                  beta1=self.config.optimizer_adam_beta1,
                                                  beta2=self.config.optimizer_adam_beta2)
        # Make the discriminator model
        ########################################################################
        # DISCRIMINATOR
        ########################################################################

        inputs_d = tf.keras.layers.Input(shape=self.config.image_dims)
        x_d = Conv2D(32, (5, 5), strides=(2, 2), padding="same",
                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), )(inputs_d)
        x_d = LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_d)
        x_d = Dropout(rate=self.config.dropout_rate)(x_d)

        x_d = Conv2D(64, (5, 5), strides=(2, 2), padding="same",
                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))(x_d)
        x_d = LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_d)
        x_d = Dropout(rate=self.config.dropout_rate)(x_d)
        x_d = Flatten()(x_d)
        x_d = Dense(1)(x_d)

        base_discriminator = Model(inputs=inputs_d, outputs=x_d)
        # Make the Generator model
        ########################################################################
        # GENERATOR
        ########################################################################

        # Input layer creates the entry point to the model
        inputs_g = Input(shape=[self.config.noise_dim])
        # Densely connected Neural Network layer with 12544 Neurons.
        x_g = Dense(7 * 7 * 256, use_bias=False,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))(inputs_g)
        # Normalize the output of the Layer
        x_g = BatchNormalization(momentum=self.config.batch_momentum)(x_g)
        # f(x) = alpha * x for x < 0, f(x) = x for x >= 0.
        x_g = LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_g)
        # Reshaping the output
        x_g = Reshape((7, 7, 256))(x_g)
        # Check the size of the current output just in case
        assert x_g.get_shape().as_list() == [None, 7, 7, 256]
        x_g = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))(x_g)
        assert x_g.get_shape().as_list() == [None, 7, 7, 128]
        x_g = BatchNormalization(momentum=self.config.batch_momentum)(x_g)
        x_g = LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_g)

        x_g = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))(x_g)
        assert x_g.get_shape().as_list() == [None, 14, 14, 64]
        x_g = BatchNormalization(momentum=self.config.batch_momentum)(x_g)
        x_g = LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_g)

        x_g = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))(x_g)
        x_g = Activation("tanh")(x_g)
        assert x_g.get_shape().as_list() == [None, 28, 28, 1]

        base_generator = tf.keras.models.Model(inputs=inputs_g, outputs=x_g)

        self.discriminator = Model(inputs=base_discriminator.inputs,outputs=base_discriminator.outputs)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])
        self.generator = Model(inputs=base_generator.inputs, outputs=base_generator.outputs)
        frozen_D = Model(inputs=base_discriminator.inputs,outputs=base_discriminator.outputs)
        frozen_D.trainable = False

        z_input = Input(shape=[self.config.noise_dim])
        img = self.generator(z_input)

        valid = frozen_D(img)

        self.combined = Model(z_input,valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def save(self, checkpoint_path):
        pass

    def load(self, checkpoint_path):
        pass
