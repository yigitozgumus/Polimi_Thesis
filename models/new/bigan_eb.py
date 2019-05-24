import tensorflow as tf

from base.base_model import BaseModel
from utils.alad_utils import get_getter


class EBBIGAN(BaseModel):
    def __init__(self, config):
        super(EBBIGAN, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # Initializations
        # Build Training Graph
        # Loss functions
        # Optimizers
        # Build Test Graph
        # Tensorboard
        pass

    def generator(self, noise_input, getter=None):
        with tf.variable_scope("Generator", custom_getter=getter, reuse=tf.AUTO_REUSE):
            net_name = "Layer_1"
            with tf.variable_scope(net_name):
                x_g = tf.layers.Dense(
                    units=4 * 4 * 512, kernel_initializer=self.init_kernel, name="fc"
                )(noise_input)
                x_g = tf.layers.batch_normalization(
                    x_g,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="batch_normalization",
                )
                x_g = tf.nn.leaky_relu(
                    features=x_g, alpha=self.config.trainer.leakyReLU_alpha, name="relu"
                )
            x_g = tf.reshape(x_g, [-1, 4, 4, 512])
            net_name = "Layer_2"
            with tf.variable_scope(net_name):
                x_g = tf.layers.Conv2DTranspose(
                    filters=512,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv2t",
                )(x_g)
                x_g = tf.layers.batch_normalization(
                    x_g,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="batch_normalization",
                )
                x_g = tf.nn.leaky_relu(
                    features=x_g, alpha=self.config.trainer.leakyReLU_alpha, name="relu"
                )
            net_name = "Layer_3"
            with tf.variable_scope(net_name):
                x_g = tf.layers.Conv2DTranspose(
                    filters=256,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv2t",
                )(x_g)
                x_g = tf.layers.batch_normalization(
                    x_g,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="batch_normalization",
                )
                x_g = tf.nn.leaky_relu(
                    features=x_g, alpha=self.config.trainer.leakyReLU_alpha, name="relu"
                )
            net_name = "Layer_4"
            with tf.variable_scope(net_name):
                x_g = tf.layers.Conv2DTranspose(
                    filters=128,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv2t",
                )(x_g)
                x_g = tf.layers.batch_normalization(
                    x_g,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="batch_normalization",
                )
                x_g = tf.nn.leaky_relu(
                    features=x_g, alpha=self.config.trainer.leakyReLU_alpha, name="relu"
                )
            net_name = "Layer_5"
            with tf.variable_scope(net_name):
                x_g = tf.layers.Conv2DTranspose(
                    filters=1,
                    kernel_size=5,
                    strides=1,
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv2t",
                )(x_g)
                x_g = tf.tanh(x_g, name="tanh")
        return x_g

    def encoder(self, image_input, getter=None):
        with tf.variable_scope("Encoder", custom_getter=getter, reuse=tf.AUTO_REUSE):
            x_e = tf.reshape(
                image_input,
                [-1, self.config.data_loader.image_size, self.config.data_loader.image_size, 1],
            )
            net_name = "Layer_1"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Conv2D(
                    filters=64,
                    kernel_size=4,
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )(x_e)
                x_e = tf.nn.leaky_relu(
                    features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
            net_name = "Layer_2"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Conv2D(
                    filters=128,
                    kernel_size=4,
                    padding="same",
                    strides=(2, 2),
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )(x_e)
                x_e = tf.layers.batch_normalization(
                    x_e, momentum=self.config.trainer.batch_momentum, training=self.is_training
                )
                x_e = tf.nn.leaky_relu(
                    features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
            net_name = "Layer_3"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Conv2D(
                    filters=256,
                    kernel_size=4,
                    padding="same",
                    strides=(2, 2),
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )(x_e)
                x_e = tf.layers.batch_normalization(
                    x_e, momentum=self.config.trainer.batch_momentum, training=self.is_training
                )
                x_e = tf.nn.leaky_relu(
                    features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
            net_name = "Layer_4"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Conv2D(
                    filters=512,
                    kernel_size=4,
                    padding="same",
                    strides=(2, 2),
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )(x_e)
                x_e = tf.layers.batch_normalization(
                    x_e, momentum=self.config.trainer.batch_momentum, training=self.is_training
                )
                x_e = tf.nn.leaky_relu(
                    features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
            net_name = "Layer_5"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Conv2D(
                    filters=self.config.trainer.noise_dim,
                    kernel_size=4,
                    padding="same",
                    strides=(2, 2),
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )(x_e)
                x_e = tf.layers.batch_normalization(
                    x_e, momentum=self.config.trainer.batch_momentum, training=self.is_training
                )
                x_e = tf.nn.leaky_relu(
                    features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
                x_e = tf.squeeze(x_e, [1, 2])
        return x_e

    def discriminator(self, image_input, getter=None):
        with tf.variable_scope("Discriminator", custom_getter=getter, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("Encoder"):
                pass
            with tf.variable_scope("Decoder"):
                pass

    def mse_loss(self, pred, data):
        loss_val = tf.sqrt(2 * tf.nn.l2_loss(pred - data)) / self.config.data_loader.batch_size
        return loss_val

    def pullaway_loss(self, embeddings):
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        similarity = tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)
        batch_size = tf.cast(tf.shape(embeddings)[0], tf.float32)
        pt_loss = (tf.reduce_sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
        return pt_loss

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.log.max_to_keep)
