import tensorflow as tf

from base.base_model import BaseModel
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation, Flatten, Dropout
from tensorflow.keras.models import Model


class GAN_keras(BaseModel):
    def __init__(self, config):
        super(GAN_keras, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # Placeholders
        self.noise_tensor = tf.placeholder(
            tf.float32, shape=[None, self.config.noise_dim], name="noise"
        )
        self.image_input = tf.placeholder(
            tf.float32, shape=[None] + self.config.image_dims, name="x"
        )
        self.random_vector_for_generation = tf.random_normal(
            [self.config.num_example_imgs_to_generate, self.config.noise_dim],
            name="sampler",
        )
        # Random Noise addition to both image and the noise
        # This makes it harder for the discriminator to do it's job, preventing
        # it from always "winning" the GAN min/max contest
        self.real_noise = tf.placeholder(
            tf.float32, shape=[None] + self.config.image_dims, name="real_noise"
        )
        self.fake_noise = tf.placeholder(
            tf.float32, shape=[None] + self.config.image_dims, name="fake_noise"
        )

        self.real_image = self.image_input + self.real_noise

        # Make the Generator model
        ########################################################################
        # GENERATOR
        ########################################################################
        with tf.variable_scope("Generator"):
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
            self.generator = tf.keras.models.Model(inputs=inputs_g, outputs=x_g)

        # Make the discriminator model
        ########################################################################
        # DISCRIMINATOR
        ########################################################################
        with tf.variable_scope("Discriminator"):
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
            self.discriminator = Model(inputs=inputs_d, outputs=x_d)

        # Store the loss values for the Tensorboard
        ########################################################################
        # TENSORBOARD
        ########################################################################
        tf.summary.scalar("Generator_Loss", self.gen_loss)
        tf.summary.scalar("Discriminator_Real_Loss", self.disc_loss_real)
        tf.summary.scalar("Real_Accuracy", self.accuracy_real)
        tf.summary.scalar("Fake_Accuracy", self.accuracy_fake)
        tf.summary.scalar("Total_Accuracy", self.accuracy_total)
        tf.summary.scalar("Discriminator_Gen_Loss", self.disc_loss_fake)
        tf.summary.scalar("Discriminator_Total_Loss", self.total_disc_loss)
        # Images for the Tensorboard
        tf.summary.image("From_Noise", tf.reshape(generated_sample, [-1, 28, 28, 1]))
        tf.summary.image("Real_Image", tf.reshape(self.image_input, [-1, 28, 28, 1]))
        # Sample Operation
        with tf.name_scope("Generator_Progress"):
            self.progress_images = self.generator(self.noise_tensor, training=False)



    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
