import tensorflow as tf

from base.base_model import BaseModel
from utils.alad_utils import get_getter
import utils.alad_utils as sn


class FenceGAN(BaseModel):
    def __init__(self, config):
        super(FenceGAN, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # Initializations
        # Kernel initialization for the convolutions
        self.init_kernel = tf.contrib.layers.xavier_initializer(
            uniform=False, seed=None, dtype=tf.dtypes.float32
        )
        # Placeholders
        self.is_training = tf.placeholder(tf.bool)
        self.image_input = tf.placeholder(
            tf.float32, shape=[None] + self.config.trainer.image_dims, name="x"
        )
        self.noise_tensor = tf.placeholder(
            tf.float32, shape=[None, self.config.trainer.noise_dim], name="noise"
        )
        # Build Training Graph
        self.logger.info("Building training graph...")
        with tf.variable_scope("FenceGAN"):
            with tf.variable_scope("Generator_Model"):
                self.image_gen = self.generator(self.noise_tensor)

            with tf.variable_scope("Discriminator_Model"):
                self.disc_real, self.f_layer_real = self.discriminator(
                    self.image_input, do_spectral_norm=self.config.trainer.do_spectral_norm
                )
                self.disc_fake, self.f_layer_fake = self.discriminator(
                    self.image_gen, do_spectral_norm=self.config.trainer.do_spectral_norm
                )
        # Loss functions
        with tf.name_scope("Loss_Functions"):
            # Discriminator Loss
            # Generator Loss

        # Optimizers
        with tf.name_scope("Optimizers"):
            self.generator_optimizer = tf.train.AdamOptimizer(
                self.config.trainer.standard_lr,
                beta1=self.config.trainer.optimizer_adam_beta1,
                beta2=self.config.trainer.optimizer_adam_beta2,
            )
            self.discriminator_optimizer = tf.train.AdamOptimizer(
                self.config.trainer.standard_lr,
                beta1=self.config.trainer.optimizer_adam_beta1,
                beta2=self.config.trainer.optimizer_adam_beta2,
            )
            # Collect all the variables
            all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # Generator Network Variables
            self.generator_vars = [
                v for v in all_variables if v.name.startswith("FenceGAN/Generator_Model")
            ]
            # Discriminator Network Variables
            self.discriminator_vars = [
                v for v in all_variables if v.name.startswith("FenceGAN/Discriminator_Model")
            ]
            # Generator Network Operations
            self.gen_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="FenceGAN/Generator_Model"
            )
            # Discriminator Network Operations
            self.disc_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="FenceGAN/Discriminator_Model"
            )
            with tf.control_dependencies(self.gen_update_ops):
                self.gen_op = self.generator_optimizer.minimize(
                    self.loss_generator,
                    var_list=self.generator_vars,
                    global_step=self.global_step_tensor,
                )
            with tf.control_dependencies(self.disc_update_ops):
                self.disc_op = self.discriminator_optimizer.minimize(
                    self.loss_discriminator, var_list=self.discriminator_vars
                )
            # Exponential Moving Average for Estimation
            self.dis_ema = tf.train.ExponentialMovingAverage(decay=self.config.trainer.ema_decay)
            maintain_averages_op_dis = self.dis_ema.apply(self.discriminator_vars)

            self.gen_ema = tf.train.ExponentialMovingAverage(decay=self.config.trainer.ema_decay)
            maintain_averages_op_gen = self.gen_ema.apply(self.generator_vars)

            with tf.control_dependencies([self.disc_op]):
                self.train_dis_op = tf.group(maintain_averages_op_dis)

            with tf.control_dependencies([self.gen_op]):
                self.train_gen_op = tf.group(maintain_averages_op_gen)

        # Build Test Graph
        self.logger.info("Building Testing Graph...")
        with tf.variable_scope("FenceGAN"):
            with tf.variable_scope("Generator_Model"):
                self.image_gen_ema = self.generator(self.noise_tensor,getter=get_getter(self.gen_ema)
)

            with tf.variable_scope("Discriminator_Model"):
                self.disc_real_ema, self.f_layer_real_ema = self.discriminator(
                    self.image_input, do_spectral_norm=self.config.trainer.do_spectral_norm
                    ,getter=get_getter(self.dis_ema)
                )
                self.disc_fake_ema, self.f_layer_fake_ema = self.discriminator(
                    self.image_gen, do_spectral_norm=self.config.trainer.do_spectral_norm
                    ,getter=get_getter(self.dis_ema)

                )
        with tf.name_scope("Testing"):
            with tf.name_scope("Image_Based"):
                delta = self.image_input - self.image_gen_ema
                delta_flat = tf.layers.Flatten()(delta)
                img_score_l1 = tf.norm(
                    delta_flat, ord=1, axis=1, keepdims=False, name="img_loss__1"
                )
                self.img_score_l1 = tf.squeeze(img_score_l1)

                delta = self.image_input - self.image_gen_ema
                delta_flat = tf.layers.Flatten()(delta)
                img_score_l2 = tf.norm(
                    delta_flat, ord=2, axis=1, keepdims=False, name="img_loss__2"
                )
                self.img_score_l2 = tf.squeeze(img_score_l2)
            with tf.name_scope("Noise_Based"):
                delta = self.embedding_rec_ema - self.embedding_q_ema
                delta_flat = tf.layers.Flatten()(delta)
                z_score_l1 = tf.norm(delta_flat, ord=1, axis=1, keepdims=False, name="z_loss_1")
                self.z_score_l1 = tf.squeeze(z_score_l1)

                delta = self.embedding_rec_ema - self.embedding_q_ema
                delta_flat = tf.layers.Flatten()(delta)
                z_score_l2 = tf.norm(delta_flat, ord=2, axis=1, keepdims=False, name="z_loss_2")
                self.z_score_l2 = tf.squeeze(z_score_l2)

        # Tensorboard
        if self.config.log.enable_summary:
            with tf.name_scope("train_summary"):
                with tf.name_scope("dis_summary"):
                    tf.summary.scalar("loss_disc", self.loss_discriminator, ["dis"])
                    tf.summary.scalar("loss_disc_real", self.disc_loss_real, ["dis"])
                    tf.summary.scalar("loss_disc_fake", self.disc_loss_fake, ["dis"])
                with tf.name_scope("gen_summary"):
                    tf.summary.scalar("loss_generator", self.loss_generator, ["gen"])
                with tf.name_scope("img_summary"):
                    tf.summary.image("input_image", self.image_input, 3, ["img"])
                    tf.summary.image("reconstructed", self.image_gen, 3, ["img"])

        self.sum_op_dis = tf.summary.merge_all("dis")
        self.sum_op_gen = tf.summary.merge_all("gen")
        self.sum_op = tf.summary.merge([self.sum_op_dis, self.sum_op_gen])
        self.sum_op_im = tf.summary.merge_all("img")

    def generator(self, noise_input, getter=None):
        with tf.variable_scope("Generator", custom_getter=getter, reuse=tf.AUTO_REUSE):
            net_name = "Layer_1"
            with tf.variable_scope(net_name):
                x_g = tf.layers.Dense(
                    units=2 * 2 * 256, kernel_initializer=self.init_kernel, name="fc"
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
            x_g = tf.reshape(x_g, [-1, 2, 2, 256])
            net_name = "Layer_2"
            with tf.variable_scope(net_name):
                x_g = tf.layers.Conv2DTranspose(
                    filters=128,
                    kernel_size=5,
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
                    filters=64,
                    kernel_size=5,
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
                    filters=32,
                    kernel_size=5,
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
                    strides=2,
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv2t",
                )(x_g)
                x_g = tf.tanh(x_g, name="tanh")
        return x_g

    def discriminator(self, image_input, getter=None, do_spectral_norm=False):
        layers = sn if do_spectral_norm else tf.layers
        with tf.variable_scope("Discriminator", custom_getter=getter, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("Encoder"):
                x_e = tf.reshape(
                    image_input,
                    [-1, self.config.data_loader.image_size, self.config.data_loader.image_size, 1],
                )
                net_name = "Layer_1"
                with tf.variable_scope(net_name):
                    x_e = layers.conv2d(
                        x_e,
                        filters=32,
                        kernel_size=5,
                        strides=2,
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="conv",
                    )
                    x_e = tf.nn.leaky_relu(
                        features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                    )
                    # 14 x 14 x 64
                net_name = "Layer_2"
                with tf.variable_scope(net_name):
                    x_e = layers.conv2d(
                        x_e,
                        filters=64,
                        kernel_size=5,
                        padding="same",
                        strides=2,
                        kernel_initializer=self.init_kernel,
                        name="conv",
                    )
                    x_e = tf.layers.batch_normalization(
                        x_e, momentum=self.config.trainer.batch_momentum, training=self.is_training
                    )
                    x_e = tf.nn.leaky_relu(
                        features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                    )
                    # 7 x 7 x 128
                net_name = "Layer_3"
                with tf.variable_scope(net_name):
                    x_e = layers.conv2d(
                        x_e,
                        filters=128,
                        kernel_size=5,
                        padding="same",
                        strides=2,
                        kernel_initializer=self.init_kernel,
                        name="conv",
                    )
                    x_e = tf.layers.batch_normalization(
                        x_e, momentum=self.config.trainer.batch_momentum, training=self.is_training
                    )
                    x_e = tf.nn.leaky_relu(
                        features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                    )
                net_name = "Layer_4"
                with tf.variable_scope(net_name):
                    x_e = layers.conv2d(
                        x_e,
                        filters=256,
                        kernel_size=5,
                        padding="same",
                        strides=2,
                        kernel_initializer=self.init_kernel,
                        name="conv",
                    )
                    x_e = tf.nn.leaky_relu(
                        features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                    )
                    feature_layer = x_e
                net_name = "Layer_4"
                with tf.variable_scope(net_name):
                    x_e = tf.layers.dropout(
                        x_e,
                        rate=self.config.trainer.dropout_rate,
                        training=self.is_training,
                        name="dropout",
                    )
                    out = tf.layers.Dense(units=1, kernel_initializer=self.init_kernel, activation=tf.nn.sigmoid,name="fc")(
                        x_e
                    )
        return out, feature_layer


    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.log.max_to_keep)
