import tensorflow as tf

from base.base_model import BaseModel
from utils.alad_utils import get_getter


class SkipGANomaly(BaseModel):
    def __init__(self, config):
        """
        Args:
            config:
        """
        super(SkipGANomaly, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # Place holders
        self.img_size = self.config.data_loader.image_size
        self.is_training = tf.placeholder(tf.bool)
        self.image_input = tf.placeholder(
            dtype=tf.float32, shape=[None] + self.config.trainer.image_dims, name="x"
        )
        self.noise_tensor = tf.placeholder(
            dtype=tf.float32, shape=[None, self.config.trainer.noise_dim], name="noise"
        )
        self.init_kernel = tf.random_normal_initializer(mean=0.0, stddev=0.02)

        #######################################################################
        # GRAPH
        ########################################################################
        self.logger.info("Building Training Graph")
        with tf.variable_scope("Skip_Ganomaly"):
            with tf.variable_scope("Generator"):
                self.reconstructed_image = self.generator(self.image_input)
            with tf.variable_scope("Discriminator"):
                self.disc_real, inter_layer_real = self.discriminator(self.image_input)
                self.disc_fake, inter_layer_fake = self.discriminator(self.reconstructed_image)
        ########################################################################
        # METRICS
        ########################################################################

        ########################################################################
        # OPTIMIZATION
        ########################################################################

        ########################################################################
        # TESTING
        ########################################################################

        ########################################################################
        # TENSORBOARD
        ########################################################################

    def generator(self, image_input, getter=None):
        # Make the generator model
        with tf.variable_scope("Generator", custom_getter=getter, reuse=tf.AUTO_REUSE):
            # Encoder Part
            with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
                model_entry = tf.reshape(image_input, [-1, self.img_size, self.img_size, 1])
                net_name = "Layer_1"
                with tf.variable_scope(net_name):
                    enc_layer_1 = tf.layers.Conv2D(
                        filters=64,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="enc_conv1",
                    )(model_entry)
                    enc_layer_1 = tf.layers.batch_normalization(
                        enc_layer_1,
                        momentum=self.config.trainer.batch_momentum,
                        training=self.is_training,
                        name="enc_bn1",
                    )
                    enc_layer_1 = tf.nn.leaky_relu(
                        enc_layer_1, alpha=self.config.trainer.leakyReLU_alpha, name="enc_lrelu_1"
                    )
                    # Current layer is  Batch_size x 16 x 16 x 64
                net_name = "Layer_2"
                with tf.variable_scope(net_name):
                    enc_layer_2 = tf.layers.Conv2D(
                        filters=128,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="enc_conv2",
                    )(enc_layer_1)
                    enc_layer_2 = tf.layers.batch_normalization(
                        enc_layer_2,
                        momentum=self.config.trainer.batch_momentum,
                        training=self.is_training,
                        name="enc_bn2",
                    )
                    enc_layer_2 = tf.nn.leaky_relu(
                        enc_layer_2, alpha=self.config.trainer.leakyReLU_alpha, name="enc_lrelu_2"
                    )
                    # Current layer is  Batch_size x 8 x 8 x 128
                net_name = "Layer_3"
                with tf.variable_scope(net_name):
                    enc_layer_3 = tf.layers.Conv2D(
                        filters=256,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="enc_conv3",
                    )(enc_layer_2)
                    enc_layer_3 = tf.layers.batch_normalization(
                        enc_layer_3,
                        momentum=self.config.trainer.batch_momentum,
                        training=self.is_training,
                        name="enc_bn3",
                    )
                    enc_layer_3 = tf.nn.leaky_relu(
                        enc_layer_3, alpha=self.config.trainer.leakyReLU_alpha, name="enc_lrelu_3"
                    )
                    # Current layer is  Batch_size x 4 x 4 x 256
                net_name = "Layer_4"
                with tf.variable_scope(net_name):
                    enc_layer_4 = tf.layers.Conv2D(
                        filters=512,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="enc_conv4",
                    )(enc_layer_3)
                    enc_layer_4 = tf.layers.batch_normalization(
                        enc_layer_4,
                        momentum=self.config.trainer.batch_momentum,
                        training=self.is_training,
                        name="enc_bn4",
                    )
                    enc_layer_4 = tf.nn.leaky_relu(
                        enc_layer_4, alpha=self.config.trainer.leakyReLU_alpha, name="enc_lrelu_4"
                    )
                    # Current layer is  Batch_size x 2 x 2 x 512
                net_name = "Layer_5"
                with tf.variable_scope(net_name):
                    enc_layer_5 = tf.layers.Conv2D(
                        filters=512,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="enc_conv5",
                    )(enc_layer_4)
            # Decoder Part
            with tf.variable_scope("Decoder", reuse=tf.AUTO_REUSE):
                gen_noise_entry = enc_layer_5
                net_name = "Layer_1"
                with tf.variable_scope(net_name):
                    dec_layer_1 = tf.layers.Conv2DTranspose(
                        filters=512,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="dec_convt1",
                    )(gen_noise_entry)
                    dec_layer_1 = tf.layers.batch_normalization(
                        dec_layer_1, momentum=self.config.trainer.batch_momentum, name="dec_bn1"
                    )
                    dec_layer_1 = tf.nn.relu(dec_layer_1, name="dec_relu1")
                    dec_layer_1 = tf.concat([enc_layer_4, dec_layer_1], axis=-1)
                    # Current layer is Batch_size x 2 x 2 x 1024
                net_name = "Layer_2"
                with tf.variable_scope(net_name):
                    dec_layer_2 = tf.layers.Conv2DTranspose(
                        filters=256,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="dec_convt2",
                    )(dec_layer_1)
                    dec_layer_2 = tf.layers.batch_normalization(
                        dec_layer_2, momentum=self.config.trainer.batch_momentum, name="dec_bn2"
                    )
                    dec_layer_2 = tf.nn.relu(dec_layer_2, name="dec_relu1")
                    dec_layer_2 = tf.concat([enc_layer_3, dec_layer_2], axis=-1)
                    # Current layer is Batch_size x 4 x 4 x 512
                net_name = "Layer_3"
                with tf.variable_scope(net_name):
                    dec_layer_3 = tf.layers.Conv2DTranspose(
                        filters=128,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="dec_convt3",
                    )(dec_layer_2)
                    dec_layer_3 = tf.layers.batch_normalization(
                        dec_layer_3, momentum=self.config.trainer.batch_momentum, name="dec_bn3"
                    )
                    dec_layer_3 = tf.nn.relu(dec_layer_3, name="dec_relu3")
                    dec_layer_3 = tf.concat([enc_layer_2, dec_layer_3], axis=-1)
                    # Current layer is Batch_size x 8 x 8 x 256
                net_name = "Layer_4"
                with tf.variable_scope(net_name):
                    dec_layer_4 = tf.layers.Conv2DTranspose(
                        filters=64,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="dec_convt4",
                    )(dec_layer_3)
                    dec_layer_4 = tf.layers.batch_normalization(
                        dec_layer_4, momentum=self.config.trainer.batch_momentum, name="dec_bn4"
                    )
                    dec_layer_4 = tf.nn.relu(dec_layer_4, name="dec_relu4")
                    dec_layer_4 = tf.concat([enc_layer_1, dec_layer_4], axis=-1)
                    # Current layer is Batch_size x 16 x 16 x 128
                net_name = "Layer_5"
                with tf.variable_scope(net_name):
                    dec_layer_5 = tf.layers.Conv2DTranspose(
                        filters=1,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="dec_convt1",
                    )(dec_layer_4)
                    # Current layer is Batch_size x 32 x 32 x 1
        return dec_layer_5

    def discriminator(self, image_input, getter=None):
        with tf.variable_scope("Discriminator", custom_getter=getter, reuse=tf.AUTO_REUSE):
            net_name = "Layer_1"
            with tf.variable_scope(net_name):
                x_d = tf.layers.Conv2D(
                    filters=64,
                    kernel_size=5,
                    strides=(1, 1),
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="d_conv1",
                )(image_input)
                x_d = tf.layers.batch_normalization(
                    inputs=x_d,
                    momentum=self.config.trainer.batch_momentum,
                    training=True,
                    name="d_bn_1",
                )
                x_d = tf.nn.leaky_relu(
                    features=x_d, alpha=self.config.trainer.leakyReLU_alpha, name="d_lr_1"
                )
            net_name = "Layer_2"
            with tf.variable_scope(net_name):
                # Second Convolutional Layer
                x_d = tf.layers.Conv2D(
                    filters=128,
                    kernel_size=5,
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="d_conv_2",
                )(x_d)
                x_d = tf.layers.batch_normalization(
                    inputs=x_d,
                    momentum=self.config.trainer.batch_momentum,
                    training=True,
                    name="d_bn_2",
                )
                x_d = tf.nn.leaky_relu(
                    features=x_d, alpha=self.config.trainer.leakyReLU_alpha, name="d_lr_2"
                )
            net_name = "Layer_3"
            with tf.variable_scope(net_name):
                # Third Convolutional Layer
                x_d = tf.layers.Conv2D(
                    filters=256,
                    kernel_size=5,
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="d_conv_3",
                )(x_d)
                x_d = tf.layers.batch_normalization(
                    inputs=x_d,
                    momentum=self.config.trainer.batch_momentum,
                    training=True,
                    name="d_bn_3",
                )
                x_d = tf.nn.leaky_relu(
                    features=x_d, alpha=self.config.trainer.leakyReLU_alpha, name="d_lr_3"
                )
            net_name = "Layer_4"
            with tf.variable_scope(net_name):
                x_d = tf.layers.Conv2D(
                    filters=512,
                    kernel_size=5,
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="d_conv_3",
                )(x_d)
                x_d = tf.layers.batch_normalization(
                    inputs=x_d,
                    momentum=self.config.trainer.batch_momentum,
                    training=True,
                    name="d_bn_3",
                )
                x_d = tf.nn.leaky_relu(
                    features=x_d, alpha=self.config.trainer.leakyReLU_alpha, name="d_lr_3"
                )
            net_name = "Layer_5"
            with tf.variable_scope(net_name):
                x_d = tf.layers.Flatten(name="d_flatten")(x_d)
                x_d = tf.layers.dropout(
                    x_d,
                    rate=self.config.trainer.dropout_rate,
                    training=self.is_training,
                    name="d_dropout",
                )
                intermediate_layer = x_d
                x_d = tf.layers.Dense(units=1, name="d_dense")(x_d)
        return x_d, intermediate_layer

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.log.max_to_keep)
