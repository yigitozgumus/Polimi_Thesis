import tensorflow as tf

from base.base_model import BaseModel
from utils.alad_utils import get_getter


class FAnogan(BaseModel):
    def __init__(self, config):
        super(FAnogan, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # Kernel initialization for the convolutions
        self.init_kernel = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        # Placeholders
        self.is_training_gen = tf.placeholder(tf.bool)
        self.is_training_dis = tf.placeholder(tf.bool)
        self.is_training_enc = tf.placeholder(tf.bool)
        self.image_input = tf.placeholder(
            tf.float32, shape=[None] + self.config.trainer.image_dims, name="x"
        )
        self.noise_tensor = tf.placeholder(
            tf.float32, shape=[None, self.config.trainer.noise_dim], name="noise"
        )
        self.true_labels = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="true_labels")
        self.generated_labels = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="gen_labels")
        self.real_noise = tf.placeholder(
            dtype=tf.float32, shape=[None] + self.config.trainer.image_dims, name="real_noise"
        )
        self.fake_noise = tf.placeholder(
            dtype=tf.float32, shape=[None] + self.config.trainer.image_dims, name="fake_noise"
        )
        self.logger.info("Building training graph...")
        with tf.variable_scope("FAnogan"):
            # Generator and Discriminator Training
            with tf.variable_scope("Generator_Model"):
                self.image_gen = self.generator(self.noise_tensor) + self.fake_noise
            with tf.variable_scope("Discriminator_Model"):
                self.disc_real, self.disc_f_real = self.discriminator(
                    self.image_input + self.real_noise
                )
                self.disc_fake, self.disc_f_fake = self.discriminator(self.image_gen)
            # Encoder Training

            with tf.variable_scope("Encoder_Model"):
                # ZIZ Architecture
                self.encoded_gen_noise = self.encoder(self.image_gen)
                # IZI Architecture
                self.encoded_img = self.encoder(self.image_input)
            with tf.variable_scope("Generator_Model"):
                self.gen_enc_img = self.generator(self.encoded_img)
            with tf.variable_scope("Discriminator_Model"):
                # IZI Training
                self.disc_real_izi, self.disc_f_real_izi = self.discriminator(self.image_input)
                self.disc_fake_izi, self.disc_f_fake_izi = self.discriminator(self.gen_enc_img)
        with tf.name_scope("Loss_Funcions"):
            with tf.name_scope("Encoder"):
                if self.config.trainer.encoder_training_mode == "ziz":
                    self.loss_encoder = self.mse_loss(
                        self.encoded_gen_noise,
                        self.noise_tensor,
                        mode=self.config.trainer.encoder_loss_mode,
                    ) * (1.0 / self.config.trainer.noise_dim)
                elif self.config.trainer.encoder_training_mode == "izi":
                    self.izi_reconstruction = self.mse_loss(
                        self.image_input,
                        self.gen_enc_img,
                        mode=self.config.trainer.encoder_loss_mode,
                    ) * (
                        1.0
                        / (self.config.data_loader.image_size * self.config.data_loader.image_size)
                    )
                    self.loss_encoder = self.izi_reconstruction
                elif self.config.trainer.encoder_training_mode == "izi_f":
                    self.izi_reconstruction = self.mse_loss(
                        self.image_input,
                        self.gen_enc_img,
                        mode=self.config.trainer.encoder_loss_mode,
                    ) * (
                        1.0
                        / (self.config.data_loader.image_size * self.config.data_loader.image_size)
                    )
                    self.izi_disc = self.mse_loss(
                        self.disc_f_real_izi,
                        self.disc_f_fake_izi,
                        mode=self.config.trainer.encoder_loss_mode,
                    ) * (
                        1.0
                        * self.config.trainer.kappa_weight_factor
                        / self.config.trainer.feature_layer_dim
                    )
                    self.loss_encoder = self.izi_reconstruction + self.izi_disc
            with tf.name_scope("Discriminator_Generator"):
                if self.config.trainer.mode == "standard":
                    self.loss_disc_real = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=self.true_labels, logits=self.disc_real
                        )
                    )
                    self.loss_disc_fake = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=self.generated_labels, logits=self.disc_fake
                        )
                    )
                    self.loss_discriminator = self.loss_disc_real + self.loss_disc_fake
                    # Flip the weigths for the encoder and generator
                    if self.config.trainer.flip_labels:
                        labels_gen = tf.zeros_like(self.disc_fake)
                    else:
                        labels_gen = tf.ones_like(self.disc_fake)
                    # Generator
                    self.loss_generator_ce = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=labels_gen, logits=self.disc_fake
                        )
                    )
                    delta = self.disc_f_fake - self.disc_f_real
                    delta = tf.layers.Flatten()(delta)
                    self.loss_generator_fm = tf.reduce_mean(
                        tf.norm(delta, ord=2, axis=1, keepdims=False)
                    )
                    self.loss_generator = (
                        self.loss_generator_ce
                        + self.config.trainer.feature_match_weight * self.loss_generator_fm
                    )
                elif self.config.trainer.mode == "wgan":
                    self.loss_d_fake = -tf.reduce_mean(self.disc_fake)
                    self.loss_d_real = -tf.reduce_mean(self.disc_real)
                    self.loss_discriminator = -self.loss_d_fake + self.loss_d_real
                    self.loss_generator = -tf.reduce_mean(self.disc_fake)

                # Weight Clipping and Encoder Part
                elif self.config.traiiner.mode == "wgan_gp":
                    self.loss_generator = -tf.reduce_mean(self.disc_fake)
                    self.loss_d_fake = -tf.reduce_mean(self.disc_fake)
                    self.loss_d_real = -tf.reduce_mean(self.disc_real)
                    self.loss_discriminator = -self.loss_d_fake - self.loss_d_real
                    alpha_x = tf.random_uniform(
                        shape=[self.config.data_loader.batch_size] + self.config.trainer.image_dims,
                        minval=0.0,
                        maxval=1.0,
                    )
                    differences_x = self.image_gen - self.image_input
                    interpolates_x = self.image_input + (alpha_x * differences_x)
                    gradients = tf.gradients(self.discriminator(interpolates_x), [interpolates_x])[
                        0
                    ]
                    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                    gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
                    self.loss_discriminator += self.config.trainer.wgan_gp_lambda * gradient_penalty
        with tf.name_scope("Optimizations"):
            if self.config.trainer.mode == "standard":
                # Build the optimizers
                self.generator_optimizer = tf.train.AdamOptimizer(
                    self.config.trainer.standard_lr,
                    beta1=self.config.trainer.optimizer_adam_beta1,
                    beta2=self.config.trainer.optimizer_adam_beta2,
                )
                self.discriminator_optimizer = tf.train.AdamOptimizer(
                    self.config.trainer.standard_lr_disc,
                    beta1=self.config.trainer.optimizer_adam_beta1,
                    beta2=self.config.trainer.optimizer_adam_beta2,
                )
                self.encoder_optimizer = tf.train.AdamOptimizer(
                    self.config.trainer.standard_lr,
                    beta1=self.config.trainer.optimizer_adam_beta1,
                    beta2=self.config.trainer.optimizer_adam_beta2,
                )
            elif self.config.trainer.mode == "wgan":
                # Build the optimizers
                self.generator_optimizer = tf.train.RMSPropOptimizer(self.config.trainer.wgan_lr)
                self.discriminator_optimizer = tf.train.RMSPropOptimizer(
                    self.config.trainer.wgan_lr
                )
                self.encoder_optimizer = tf.train.AdamOptimizer(
                    self.config.trainer.wgan_lr,
                    beta1=self.config.trainer.optimizer_adam_beta1,
                    beta2=self.config.trainer.optimizer_adam_beta2,
                )
            elif self.config.trainer.mode == "wgan-gp":
                # Build the optimizers
                self.generator_optimizer = tf.train.AdamOptimizer(
                    self.config.trainer.wgan_gp_lr, beta1=0.0, beta2=0.9
                )
                self.discriminator_optimizer = tf.train.AdamOptimizer(
                    self.config.trainer.wgan_gp_lr, beta1=0.0, beta2=0.9
                )
                self.encoder_optimizer = tf.train.AdamOptimizer(
                    self.config.trainer.wgan_gp_lr, beta1=0.0, beta2=0.9
                )
            # Collect all the variables
            all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # Generator Network Variables
            self.generator_vars = [
                v for v in all_variables if v.name.startswith("FAnogan/Generator_Model")
            ]
            # Discriminator Network Variables
            self.discriminator_vars = [
                v for v in all_variables if v.name.startswith("FAnogan/Discriminator_Model")
            ]
            if self.config.trainer.mode == "wgan":
                clip_ops = []
                for var in self.discriminator_vars:
                    clip_bounds = [-0.01, 0.01]
                    clip_ops.append(
                        tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1]))
                    )
                self.clip_disc_weights = tf.group(*clip_ops)
            # Encoder Network Variables
            self.encoder_vars = [
                v for v in all_variables if v.name.startswith("FAnogan/Encoder_Model")
            ]
            # Create Training Operations
            # Generator Network Operations
            self.gen_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="FAnogan/Generator_Model"
            )
            # Discriminator Network Operations
            self.disc_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="FAnogan/Discriminator_Model"
            )
            # Encoder Network Operations
            self.enc_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="FAnogan/Encoder_Model"
            )
            # Initialization of Optimizers
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
            with tf.control_dependencies(self.enc_update_ops):
                self.enc_op = self.encoder_optimizer.minimize(
                    self.loss_encoder, var_list=self.encoder_vars
                )
            # Exponential Moving Average for Estimation
            self.dis_ema = tf.train.ExponentialMovingAverage(decay=self.config.trainer.ema_decay)
            maintain_averages_op_dis = self.dis_ema.apply(self.discriminator_vars)

            self.gen_ema = tf.train.ExponentialMovingAverage(decay=self.config.trainer.ema_decay)
            maintain_averages_op_gen = self.gen_ema.apply(self.generator_vars)

            self.enc_ema = tf.train.ExponentialMovingAverage(decay=self.config.trainer.ema_decay)
            maintain_averages_op_enc = self.enc_ema.apply(self.encoder_vars)

            with tf.control_dependencies([self.disc_op]):
                self.train_dis_op = tf.group(maintain_averages_op_dis)

            with tf.control_dependencies([self.gen_op]):
                self.train_gen_op = tf.group(maintain_averages_op_gen)

            with tf.control_dependencies([self.enc_op]):
                self.train_enc_op = tf.group(maintain_averages_op_enc)

        self.logger.info("Building Testing Graph...")

        with tf.variable_scope("FAnogan"):
            # Generator and Discriminator Training
            with tf.variable_scope("Generator_Model"):
                self.image_gen_ema = self.generator(
                    self.noise_tensor, getter=get_getter(self.gen_ema)
                )
            with tf.variable_scope("Discriminator_Model"):
                self.disc_real_ema, self.disc_f_real_ema = self.discriminator(
                    self.image_input, getter=get_getter(self.dis_ema)
                )
                self.disc_fake_ema, self.disc_f_fake_ema = self.discriminator(
                    self.image_gen_ema, getter=get_getter(self.dis_ema)
                )
            # Encoder Training

            with tf.variable_scope("Encoder_Model"):
                # ZIZ Architecture
                self.encoded_gen_noise_ema = self.encoder(
                    self.image_gen_ema, getter=get_getter(self.enc_ema)
                )
                # IZI Architecture
                self.encoded_img_ema = self.encoder(
                    self.image_input, getter=get_getter(self.enc_ema)
                )
            with tf.variable_scope("Generator_Model"):
                self.gen_enc_img_ema = self.generator(
                    self.encoded_img_ema, getter=get_getter(self.gen_ema)
                )
            with tf.variable_scope("Discriminator_Model"):
                # IZI Training
                self.disc_real_izi_ema, self.disc_f_real_izi_ema = self.discriminator(
                    self.image_input, getter=get_getter(self.dis_ema)
                )
                self.disc_fake_izi_ema, self.disc_f_fake_izi_ema = self.discriminator(
                    self.gen_enc_img_ema, getter=get_getter(self.dis_ema)
                )

        with tf.name_scope("Testing"):
            with tf.name_scope("izi_f_loss"):
                self.score_reconstruction = self.mse_loss(
                    self.image_input, self.gen_enc_img_ema
                ) * (
                    1.0 / (self.config.data_loader.image_size * self.config.data_loader.image_size)
                )
                self.score_disc = self.mse_loss(
                    self.disc_f_real_izi_ema, self.disc_f_fake_izi_ema
                ) * (
                    1.0
                    * self.config.trainer.kappa_weight_factor
                    / self.config.trainer.feature_layer_dim
                )
                self.izi_f_score = self.score_reconstruction + self.score_disc
            with tf.name_scope("ziz_loss"):
                self.score_reconstruction = self.mse_loss(
                    self.image_input, self.gen_enc_img_ema
                ) * (
                    1.0 / (self.config.data_loader.image_size * self.config.data_loader.image_size)
                )
                self.ziz_score = self.score_reconstruction

        if self.config.trainer.enable_early_stop:
            self.rec_error_valid = tf.reduce_mean(self.izi_f_score)

        if self.config.log.enable_summary:
            with tf.name_scope("Summary"):
                with tf.name_scope("Disc_Summary"):
                    tf.summary.scalar("loss_discriminator", self.loss_discriminator, ["dis"])
                    if self.config.trainer.mode == "standard":
                        tf.summary.scalar("loss_dis_real", self.loss_disc_real, ["dis"])
                        tf.summary.scalar("loss_dis_fake", self.loss_disc_fake, ["dis"])
                with tf.name_scope("Gen_Summary"):
                    tf.summary.scalar("loss_generator", self.loss_generator, ["gen"])
                    if self.config.trainer.mode == "standard":
                        tf.summary.scalar("loss_generator_ce", self.loss_generator_ce, ["gen"])
                        tf.summary.scalar("loss_generator_fm", self.loss_generator_fm, ["gen"])
                    tf.summary.scalar("loss_encoder", self.loss_encoder, ["enc"])
                with tf.name_scope("Image_Summary"):
                    tf.summary.image("reconstruct", self.image_gen, 3, ["image_1"])
                    tf.summary.image("input_images", self.image_input, 3, ["image_1"])
                    tf.summary.image("gen_enc_img", self.gen_enc_img, 3, ["image_2"])
                    tf.summary.image("input_image_2", self.image_input, 3, ["image_2"])
        if self.config.trainer.enable_early_stop:
            with tf.name_scope("validation_summary"):
                tf.summary.scalar("valid", self.rec_error_valid, ["v"])

        self.sum_op_dis = tf.summary.merge_all("dis")
        self.sum_op_gen = tf.summary.merge_all("gen")
        self.sum_op_enc = tf.summary.merge_all("enc")
        self.sum_op_im_1 = tf.summary.merge_all("image_1")
        self.sum_op_im_2 = tf.summary.merge_all("image_2")
        self.sum_op_valid = tf.summary.merge_all("v")

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
                    training=self.is_training_gen,
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
                    training=self.is_training_gen,
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
                    training=self.is_training_gen,
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
                    training=self.is_training_gen,
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

    def discriminator(self, image_input, getter=None):
        with tf.variable_scope("Discriminator", custom_getter=getter, reuse=tf.AUTO_REUSE):
            # First Convolutional Layer
            net_name = "Layer_1"
            with tf.variable_scope(net_name):
                x_d = tf.layers.Conv2D(
                    filters=64,
                    kernel_size=5,
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="d_conv1",
                )(image_input)
                x_d = tf.layers.batch_normalization(
                    inputs=x_d,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training_dis,
                    name="d_bn_1",
                )
                x_d = tf.nn.leaky_relu(
                    features=x_d, alpha=self.config.trainer.leakyReLU_alpha, name="d_lr_1"
                )  # 28 x 28 x 64
            # Second Convolutional Layer
            net_name = "Layer_2"
            with tf.variable_scope(net_name):
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
                    training=self.is_training_dis,
                    name="d_bn_2",
                )
                x_d = tf.nn.leaky_relu(
                    features=x_d, alpha=self.config.trainer.leakyReLU_alpha, name="d_lr_2"
                )  # 14 x 14 x 128
            # Third Convolutional Layer
            net_name = "Layer_3"
            with tf.variable_scope(net_name):
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
                    training=self.is_training_dis,
                    name="d_bn_3",
                )
                x_d = tf.nn.leaky_relu(
                    features=x_d, alpha=self.config.trainer.leakyReLU_alpha, name="d_lr_3"
                )  # 7 x 7 x 256
            net_name = "Layer_4"
            with tf.variable_scope(net_name):
                x_d = tf.layers.Flatten(name="d_flatten")(x_d)
                x_d = tf.layers.batch_normalization(
                    inputs=x_d,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training_dis,
                    name="d_bn_4",
                )
                x_d = tf.nn.leaky_relu(
                    features=x_d, alpha=self.config.trainer.leakyReLU_alpha, name="d_lr_4"
                )
            intermediate_layer = x_d
            net_name = "Layer_5"
            with tf.variable_scope(net_name):
                x_d = tf.layers.Dense(units=1, name="d_dense")(x_d)
            return x_d, intermediate_layer

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
                    kernel_size=5,
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )(x_e)
                x_e = tf.layers.batch_normalization(
                    x_e, momentum=self.config.trainer.batch_momentum, training=self.is_training_enc
                )

                x_e = tf.nn.leaky_relu(
                    features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
            net_name = "Layer_2"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Conv2D(
                    filters=128,
                    kernel_size=5,
                    padding="same",
                    strides=(2, 2),
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )(x_e)
                x_e = tf.layers.batch_normalization(
                    x_e, momentum=self.config.trainer.batch_momentum, training=self.is_training_enc
                )
                x_e = tf.nn.leaky_relu(
                    features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
            net_name = "Layer_3"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Conv2D(
                    filters=256,
                    kernel_size=5,
                    padding="same",
                    strides=(2, 2),
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )(x_e)
                x_e = tf.layers.batch_normalization(
                    x_e, momentum=self.config.trainer.batch_momentum, training=self.is_training_enc
                )
                x_e = tf.nn.leaky_relu(
                    features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
            net_name = "Layer_4"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Conv2D(
                    filters=512,
                    kernel_size=5,
                    padding="same",
                    strides=(2, 2),
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )(x_e)
                x_e = tf.layers.batch_normalization(
                    x_e, momentum=self.config.trainer.batch_momentum, training=self.is_training_enc
                )
                x_e = tf.nn.leaky_relu(
                    features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
                x_e = tf.layers.Flatten()(x_e)
            net_name = "Layer_5"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Dense(
                    units=self.config.trainer.noise_dim,
                    kernel_initializer=self.init_kernel,
                    name="fc",
                )(x_e)
        return x_e

    def mse_loss(self, pred, data, mode="norm"):
        if mode == "norm":
            delta = pred - data
            delta = tf.layers.Flatten()(delta)
            loss_val = (tf.norm(delta, ord=2, axis=1, keepdims=False))
        elif mode == "mse":
            loss_val = tf.sqrt(2 * tf.nn.l2_loss(pred - data)) / self.config.data_loader.batch_size
        return loss_val

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.log.max_to_keep)
