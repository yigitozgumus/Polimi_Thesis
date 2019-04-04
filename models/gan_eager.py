from base.base_model_eager import BaseModelEager
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense, Reshape, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, Dropout, Activation, Conv2D, Flatten
from tensorflow.keras.models import Model

class GAN_eager(BaseModelEager):
    def __init__(self, config):
        super(GAN_eager, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        # Initializer for the kernels
        self.kernel_initializer = tf.random_normal_initializer(stddev=0.02)
        # Make the Generator model
        ########################################################################
        # GENERATOR
        ########################################################################

        input_g = Input(shape=[self.config.noise_dim,])
        layer_g = Dense(7 * 7 * 256, activation="relu",use_bias=False,
        kernel_initializer=self.kernel_initializer)(input_g)
        layer_g = BatchNormalization(momentum=self.config.batch_momentum)(layer_g)
        layer_g = LeakyReLU(alpha=self.config.leakyReLU_alpha)(layer_g)
        layer_g = Reshape((7, 7, 256))(layer_g)

        layer_g = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False,
        kernel_initializer=self.kernel_initializer)(layer_g)
        layer_g = BatchNormalization(momentum=self.config.batch_momentum)(layer_g)
        layer_g = LeakyReLU(alpha=self.config.leakyReLU_alpha)(layer_g)

        layer_g = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", use_bias=False,
                                  kernel_initializer=self.kernel_initializer)(layer_g)
        layer_g = BatchNormalization(momentum=self.config.batch_momentum)(layer_g)
        layer_g = LeakyReLU(alpha=self.config.leakyReLU_alpha)(layer_g)

        layer_g = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", use_bias=False,
                                  kernel_initializer=self.kernel_initializer)(layer_g)
        layer_g = BatchNormalization(momentum=self.config.batch_momentum)(layer_g)
        layer_g = LeakyReLU(alpha=self.config.leakyReLU_alpha)(layer_g)

        layer_g = Conv2DTranspose(1, (5, 5), strides=(1, 1), padding="same", use_bias=False,
                                  kernel_initializer=self.kernel_initializer)(layer_g)
        layer_g = BatchNormalization(momentum=self.config.batch_momentum)(layer_g)
        layer_g = LeakyReLU(alpha=self.config.leakyReLU_alpha)(layer_g)
        layer_g = Activation("tanh")(layer_g)

        self.generator = Model(inputs=input_g, outputs=layer_g)


        # Make the discriminator model
        ########################################################################
        # DISCRIMINATOR
        ########################################################################

        inputs_d = Input(shape=self.config.image_dims)
        # First Convolutional Layer
        x_d = Conv2D(128, (5, 5), strides=(1, 1), padding="same",
                    kernel_initializer=self.kernel_initializer)(inputs_d)
        # x_d = tf.keras.layers.BatchNormalization(momentum=self.config.batch_momentum)(x_d)
        x_d = LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_d)
        x_d = Dropout(rate=self.config.dropout_rate)(x_d)
        # Second Convolutional Layer
        x_d = Conv2D(64, (5, 5), strides=(2, 2), padding="same",
                        kernel_initializer=self.kernel_initializer)(x_d)
        # x_d = tf.keras.layers.BatchNormalization(momentum=self.config.batch_momentum)(x_d)
        x_d = LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_d)
        x_d = Dropout(rate=self.config.dropout_rate)(x_d)
        # Third Convolutional Layer
        x_d = Conv2D(32, (5, 5), strides=(2, 2), padding="same",
                        kernel_initializer=self.kernel_initializer)(x_d)
        # x_d = tf.keras.layers.BatchNormalization(momentum=self.config.batch_momentum)(x_d)
        x_d = LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_d)
        x_d = Dropout(rate=self.config.dropout_rate)(x_d)
        x_d = Flatten()(x_d)
        # x_d = tf.keras.layers.Dropout(rate=self.config.dropout_rate)(x_d)
        x_d = Dense(1)(x_d)
        self.discriminator = Model(inputs=inputs_d, outputs=x_d)

        ########################################################################
        # OPTIMIZATION
        ########################################################################
        # Build the Optimizers
        self.generator_optimizer = tf.train.AdamOptimizer(
            self.config.generator_l_rate,
            beta1=self.config.optimizer_adam_beta1,
            beta2=self.config.optimizer_adam_beta2
        )
        self.discriminator_optimizer = tf.train.AdamOptimizer(
            self.config.discriminator_l_rate,
            beta1=self.config.optimizer_adam_beta1,
            beta2=self.config.optimizer_adam_beta2
        )

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)

        




