import tensorflow as tf

from base.base_model import BaseModel
from utils.alad_utils import get_getter


class BIGAN(BaseModel):
    def __init__(self, config):
        """
        Args:
            config:
        """
        super(BIGAN, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # Kernel initialization for the convolutions
        self.init_kernel = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        # Placeholders
        self.is_training = tf.placeholder(tf.bool)
        self.image_input = tf.placeholder(
            tf.float32, shape=[None] + self.config.trainer.image_dims, name="x"
        )
        self.noise_tensor = tf.placeholder(
            tf.float32, shape=[None, self.config.trainer.noise_dim], name="noise"
        )
        self.logger.info("Building training graph...")
        with tf.variable_scope("BIGAN"):
            with tf.variable_scope("Encoder_Model"):
                self.noise_gen = self.encoder(self.image_input)
            with tf.variable_scope("Generator_Model"):
                self.image_gen = self.decoder(self.noise_tensor)
                self.reconstructed = self.decoder(self.noise_gen)
            with tf.variable_scope("Discriminator_Model"):
                l_encoder, inter_layer_inp = self.discriminator(
                    self.noise_gen, self.image_input
                )
                l_generator, inter_layer_rct = self.discriminator(
                    self.noise_tensor, self.image_gen
                )

        # Loss Function Implementations
        with tf.name_scope("Loss_Functions"):
            # Discriminator
            self.loss_dis_enc = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(l_encoder), logits=l_encoder
                )
            )
            self.loss_dis_gen = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(l_generator), logits=l_generator
                )
            )
            self.loss_discriminator = self.loss_dis_enc + self.loss_dis_gen
            # Generator
            self.loss_generator = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(l_generator), logits=l_generator
                )
            )
            # Encoder
            self.loss_encoder = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(l_encoder), logits=l_encoder
                )
            )
        # Optimizer Implementations
        with tf.name_scope("Optimizers"):
            # Build the optimizers
            # Build the Optimizers
            self.generator_optimizer = tf.train.AdamOptimizer(
                self.config.trainer.generator_l_rate,
                beta1=self.config.trainer.optimizer_adam_beta1,
                beta2=self.config.trainer.optimizer_adam_beta2,
            )
            self.discriminator_optimizer = tf.train.AdamOptimizer(
                self.config.trainer.discriminator_l_rate,
                beta1=self.config.trainer.optimizer_adam_beta1,
                beta2=self.config.trainer.optimizer_adam_beta2,
            )
            self.encoder_optimizer = tf.train.AdamOptimizer(
                self.config.trainer.discriminator_l_rate,
                beta1=self.config.trainer.optimizer_adam_beta1,
                beta2=self.config.trainer.optimizer_adam_beta2,
            )
            # Collect all the variables
            all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # Generator Network Variables
            self.generator_vars = [
                v for v in all_variables if v.name.startswith("BIGAN/Generator_Model")
            ]
            # Discriminator Network Variables
            self.discriminator_vars = [
                v
                for v in all_variables
                if v.name.startswith("BIGAN/Discriminator_Model")
            ]
            # Encoder Network Variables
            self.encoder_vars = [
                v for v in all_variables if v.name.startswith("BIGAN/Encoder_Model")
            ]
            # Create Training Operations
            # Generator Network Operations
            self.gen_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="BIGAN/Generator_Model"
            )
            # Discriminator Network Operations
            self.disc_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="BIGAN/Discriminator_Model"
            )
            # Encoder Network Operations
            self.enc_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="BIGAN/Encoder_Model"
            )
            # Initialization of Optimizers
            with tf.control_dependencies(self.gen_update_ops):
                self.gen_op = self.generator_optimizer.minimize(
                    self.loss_generator, var_list=self.generator_vars
                )
            with tf.control_dependencies(self.disc_update_ops):
                self.disc_op = self.discriminator_optimizer.minimize(
                    self.loss_discriminator, var_list=self.discriminator_vars
                )
            with tf.control_dependencies(self.enc_update_ops):
                self.enc_op = self.encoder_optimizer.minimize(
                    self.loss_encoder, var_list=self.encoder_vars
                )
            # Exponential Moving Average for Estimation
            self.dis_ema = tf.train.ExponentialMovingAverage(
                decay=self.config.trainer.ema_decay
            )
            maintain_averages_op_dis = self.dis_ema.apply(self.discriminator_vars)

            self.gen_ema = tf.train.ExponentialMovingAverage(
                decay=self.config.trainer.ema_decay
            )
            maintain_averages_op_gen = self.gen_ema.apply(self.generator_vars)

            self.enc_ema = tf.train.ExponentialMovingAverage(
                decay=self.config.trainer.ema_decay
            )
            maintain_averages_op_enc = self.enc_ema.apply(self.encoder_vars)

            with tf.control_dependencies([self.disc_op]):
                self.train_dis_op = tf.group(maintain_averages_op_dis)

            with tf.control_dependencies([self.gen_op]):
                self.train_gen_op = tf.group(maintain_averages_op_gen)

            with tf.control_dependencies([self.enc_op]):
                self.train_enc_op = tf.group(maintain_averages_op_enc)

        with tf.name_scope("Summary"):
            with tf.name_scope("Disc_Summary"):
                tf.summary.scalar(
                    "loss_discriminator", self.loss_discriminator, ["dis"]
                )
                tf.summary.scalar("loss_dis_encoder", self.loss_dis_enc, ["dis"])
                tf.summary.scalar("loss_dis_gen", self.loss_dis_gen, ["dis"])
            with tf.name_scope("Gen_Summary"):
                tf.summary.scalar("loss_generator", self.loss_generator, ["gen"])
                tf.summary.scalar("loss_encoder", self.loss_encoder, ["gen"])
            with tf.name_scope("Image_Summary"):
                tf.summary.image("reconstruct", self.reconstructed, 3, ["image"])
                tf.summary.image("input_images", self.image_input, 3, ["image"])

        self.sum_op_dis = tf.summary.merge_all("dis")
        self.sum_op_gen = tf.summary.merge_all("gen")
        self.sum_op_im = tf.summary.merge_all("image")

        self.logger.info("Building Testing Graph...")

        with tf.variable_scope("BIGAN"):
            with tf.variable_scope("Encoder_Model"):
                self.noise_gen_ema = self.encoder(
                    self.image_input, getter=get_getter(self.enc_ema)
                )
            with tf.variable_scope("Generator_Model"):
                self.reconstruct_ema = self.decoder(
                    self.noise_gen_ema, getter=get_getter(self.gen_ema)
                )
            with tf.variable_scope("Discriminator_Model"):
                self.l_encoder_ema, self.inter_layer_inp_ema = self.discriminator(
                    self.noise_gen_ema,
                    self.image_input,
                    getter=get_getter(self.dis_ema),
                )
                self.l_generator_ema, self.inter_layer_rct_ema = self.discriminator(
                    self.noise_gen_ema,
                    self.reconstruct_ema,
                    getter=get_getter(self.dis_ema),
                )

        with tf.name_scope("Testing"):
            with tf.variable_scope("Reconstruction_Loss"):
                delta = self.image_input - self.reconstruct_ema
                delta_flat = tf.layers.Flatten()(delta)
                self.gen_score = tf.norm(
                    delta_flat,
                    ord=self.config.trainer.degree,
                    axis=1,
                    keepdims=False,
                    name="epsilon",
                )
            with tf.variable_scope("Discriminator_Loss"):
                if self.config.trainer.loss_method == "cross_e":
                    self.dis_score = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.ones_like(self.l_generator_ema),
                        logits=self.l_generator_ema,
                    )
                elif self.config.trainer.loss_method == "fm":
                    fm = self.inter_layer_inp_ema - self.inter_layer_rct_ema
                    fm = tf.layers.Flatten()(fm)
                    self.dis_score = tf.norm(
                        fm,
                        ord=self.config.trainer.degree,
                        axis=1,
                        keepdims=False,
                        name="d_loss",
                    )
                self.dis_score = tf.squeeze(self.dis_score)
            with tf.variable_scope("Score"):
                self.list_scores = (
                    1 - self.config.trainer.weight
                ) * self.gen_score + self.config.trainer.weight * self.dis_score

    def encoder(self, image_input, getter=None):
        """Encoder architecture in tensorflow Maps the data into the latent

        Args:
            image_input:
            getter:

        Returns:
            (tensor): last activation layer of the encoder
        """
        with tf.variable_scope("Encoder", custom_getter=getter, reuse=tf.AUTO_REUSE):
            x_e = tf.reshape(image_input, [-1, 32, 32, 1])
            net_name = "Layer_1"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Conv2D(
                    filters=32,
                    kernel_size=3,
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )(x_e)
                x_e = tf.nn.leaky_relu(
                    x_e, alpha=self.config.trainer.leaky_ReLU_alpha, name="leaky_relu"
                )
            net_name = "Layer_2"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    padding="same",
                    strides=(2, 2),
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )(x_e)
                x_e = tf.layers.batch_normalization(
                    x_e,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                )
                x_e = tf.nn.leaky_relu(
                    x_e, alpha=self.config.trainer.leaky_ReLU_alpha, name="leaky_relu"
                )
            net_name = "Layer_3"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Conv2D(
                    filters=128,
                    kernel_size=3,
                    padding="same",
                    strides=(2, 2),
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )(x_e)
                x_e = tf.layers.batch_normalization(
                    x_e,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                )
                x_e = tf.nn.leaky_relu(
                    x_e, alpha=self.config.trainer.leaky_ReLU_alpha, name="leaky_relu"
                )
            x_e = tf.layers.Flatten()(x_e)
            net_name = "Layer_4"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Dense(
                    units=self.config.trainer.noise_dim,
                    kernel_initializer=self.init_kernel,
                    name="fc",
                )(x_e)
        return x_e

    def decoder(self, noise_input, getter=None):
        """Decoder architecture in tensorflow Generates data from the latent
        space

        Args:
            noise_input:
            getter:

        Returns:
            (tensor): last activation layer of the generator
        """
        with tf.variable_scope("Generator", custom_getter=getter, reuse=tf.AUTO_REUSE):
            net_name = "Layer_1"
            with tf.variable_scope(net_name):
                x_g = tf.layers.Dense(
                    units=1024, kernel_initializer=self.init_kernel, name="fc"
                )(noise_input)
                x_g = tf.layers.batch_normalization(
                    x_g,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="batch_normalization",
                )
                x_g = tf.nn.relu(x_g, name="relu")
            net_name = "Layer_2"
            with tf.variable_scope(net_name):
                x_g = tf.layers.Dense(
                    units=4 * 4 * 128, kernel_initializer=self.init_kernel, name="fc"
                )(x_g)
                x_g = tf.layers.batch_normalization(
                    x_g,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="batch_normalization",
                )
                x_g = tf.nn.relu(x_g, name="relu")
            x_g = tf.reshape(x_g, [-1, 4, 4, 128])
            net_name = "Layer_3"
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
                x_g = tf.nn.relu(x_g, name="relu")
            net_name = "Layer_4"
            with tf.variable_scope(net_name):
                x_g = tf.layers.Conv2DTranspose(
                    filters=64,
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
                x_g = tf.nn.relu(x_g, name="relu")
            net_name = "Layer_5"
            with tf.variable_scope(net_name):
                x_g = tf.layers.Conv2DTranspose(
                    filters=1,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv2t",
                )(x_g)
                x_g = tf.tanh(x_g, name="tanh")
        return x_g

    def discriminator(self, noise_input, image_input, getter=None):
        """ Discriminator architecture in tensorflow
        Discriminates between pairs (E(x), x) and (z, G(z))
        Args:
            noise_input:
            image_input:
            getter:
        """
        with tf.variable_scope(
            "Discriminator", custom_getter=getter, reuse=tf.AUTO_REUSE
        ):
            # D(x)
            image = tf.reshape(image_input, [-1, 32, 32, 1])
            net_name = "X_Layer_1"
            with tf.variable_scope(net_name):
                x_d = tf.layers.Conv2D(
                    filters=64,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )(image)
                x_d = tf.nn.leaky_relu(
                    x_d, alpha=self.config.trainer.leaky_ReLU_alpha, name="leaky_relu"
                )
                x_d = tf.layers.dropout(
                    x_d,
                    rate=self.config.trainer.dropout_rate,
                    name="dropout",
                    training=self.is_training,
                )
            net_name = "X_Layer_2"
            with tf.variable_scope(net_name):
                x_d = tf.layers.Conv2D(
                    filters=64,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )(x_d)
                x_d = tf.nn.leaky_relu(
                    x_d, alpha=self.config.trainer.leaky_ReLU_alpha, name="leaky_relu"
                )
                x_d = tf.layers.dropout(
                    x_d,
                    rate=self.config.trainer.dropout_rate,
                    name="dropout",
                    training=self.is_training,
                )
            x_d = tf.reshape(x_d, [-1, 8 * 8 * 64])

            # D(z)
            net_name = "Z_Layer_1"
            with tf.variable_scope(net_name):
                z = tf.layers.Dense(
                    units=512, kernel_initializer=self.init_kernel, name="fc"
                )(noise_input)
                z = tf.nn.leaky_relu(
                    z, alpha=self.config.trainer.leaky_ReLU_alpha, name="leaky_relu"
                )
                z = tf.layers.dropout(
                    z,
                    rate=self.config.trainer.dropout_rate,
                    training=self.is_training,
                    name="dropout",
                )

            # D(x, z)
            y = tf.concat([x_d, z], axis=1)

            net_name = "Y_Layer_1"
            with tf.variable_scope(net_name):
                y = tf.layers.Dense(
                    units=self.config.trainer.dis_inter_layer_dim,
                    kernel_initializer=self.init_kernel,
                    name="fc",
                )(y)
                y = tf.nn.leaky_relu(
                    y, alpha=self.config.trainer.leaky_ReLU_alpha, name="leaky_relu"
                )
                y = tf.layers.dropout(
                    y,
                    rate=self.config.trainer.dropout_rate,
                    training=self.is_training,
                    name="dropout",
                )
            intermediate_layer = y

            net_name = "y_fc_logits"
            with tf.variable_scope(net_name):
                logits = tf.layers.Dense(
                    units=1, kernel_initializer=self.init_kernel, name="fc"
                )(y)

        return logits, intermediate_layer

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.log.max_to_keep)
