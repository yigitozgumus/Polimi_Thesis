import tensorflow as tf

from base.base_model import BaseModel
from utils.alad_utils import get_getter


class GANomaly(BaseModel):
    def __init__(self, config):
        """
        Args:
            config:
        """
        super(GANomaly, self).__init__(config)
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

        self.true_labels = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="true_labels")
        self.generated_labels = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="gen_labels")
        self.real_noise = tf.placeholder(
            dtype=tf.float32, shape=[None] + self.config.trainer.image_dims, name="real_noise"
        )
        self.fake_noise = tf.placeholder(
            dtype=tf.float32, shape=[None] + self.config.trainer.image_dims, name="fake_noise"
        )
        self.logger.info("Building training graph...")

        with tf.variable_scope("GANomaly"):
            with tf.variable_scope("Generator_Model"):
                self.noise_gen, self.img_rec, self.noise_rec = self.generator(self.image_input)
                self.img_rec += self.fake_noise
            with tf.variable_scope("Discriminator_Model"):
                l_real, inter_layer_inp = self.discriminator(self.image_input + self.real_noise)
                l_fake, inter_layer_rct = self.discriminator(self.img_rec)

        with tf.name_scope("Loss_Functions"):
            # Discriminator
            self.loss_dis_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.true_labels, logits=l_real)
            )
            self.loss_dis_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.generated_labels, logits=l_fake)
            )
            # Feature matching part
            fm = inter_layer_inp - inter_layer_rct
            fm = tf.layers.Flatten()(fm)
            self.feature_match = tf.reduce_mean(tf.norm(fm, ord=2, axis=1, keepdims=False))
            self.loss_discriminator = (
                self.loss_dis_fake + self.loss_dis_real + self.feature_match
                if self.config.trainer.loss_method == "fm"
                else self.loss_dis_fake + self.loss_dis_real
            )
            # Generator
            # Adversarial Loss
            if self.config.trainer.flip_labels:
                labels = tf.zeros_like(l_fake)
            else:
                labels = tf.ones_like(l_real)
            self.gen_loss_ce = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=l_fake)
            )
            # Contextual Loss
            l1_norm = self.image_input - self.img_rec
            l1_norm = tf.layers.Flatten()(l1_norm)
            self.gen_loss_con = tf.reduce_mean(tf.norm(l1_norm, ord=1, axis=1, keepdims=False))
            # Encoder Loss
            l2_norm = self.noise_gen - self.noise_rec
            l2_norm = tf.layers.Flatten()(l2_norm)
            self.gen_loss_enc = tf.reduce_mean(tf.norm(l2_norm, ord=2, axis=1, keepdims=False))

            self.gen_loss_total = (
                self.config.trainer.weight_adv * self.gen_loss_ce
                + self.config.trainer.weight_cont * self.gen_loss_con
                + self.config.trainer.weight_enc * self.gen_loss_enc
            )

        with tf.name_scope("Optimizers"):
            # Build the optimizers
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
            # Collect all the variables
            all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # Generator Network Variables
            self.generator_vars = [
                v for v in all_variables if v.name.startswith("GANomaly/Generator_Model")
            ]
            # Discriminator Network Variables
            self.discriminator_vars = [
                v for v in all_variables if v.name.startswith("GANomaly/Discriminator_Model")
            ]
            # Create Training Operations
            # Generator Network Operations
            self.gen_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="GANomaly/Generator_Model"
            )
            # Discriminator Network Operations
            self.disc_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="GANomaly/Discriminator_Model"
            )
            # Initialization of Optimizers
            with tf.control_dependencies(self.gen_update_ops):
                self.gen_op = self.generator_optimizer.minimize(
                    self.gen_loss_total,
                    global_step=self.global_step_tensor,
                    var_list=self.generator_vars,
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

        self.logger.info("Building Testing Graph...")
        with tf.variable_scope("GANomaly"):
            with tf.variable_scope("Generator_Model"):
                self.noise_gen_ema, self.img_rec_ema, self.noise_rec_ema = self.generator(
                    self.image_input, getter=get_getter(self.gen_ema)
                )
            with tf.variable_scope("Discriminator_model"):
                self.l_real_ema, self.inter_layer_inp_ema = self.discriminator(
                    self.image_input, getter=get_getter(self.dis_ema)
                )
                self.l_fake_ema, self.inter_layer_rct_ema = self.discriminator(
                    self.img_rec_ema, getter=get_getter(self.dis_ema)
                )

        with tf.name_scope("Testing"):
            with tf.variable_scope("Reconstruction_Loss"):
                # | G_E(x) - E(G(x))|1
                # Difference between the noise generated from the input image and reconstructed noise
                delta = self.noise_gen_ema - self.noise_rec_ema
                delta = tf.layers.Flatten()(delta)
                self.score = tf.norm(delta, ord=1, axis=1, keepdims=False)

        if self.config.trainer.enable_early_stop:
            self.rec_error_valid = tf.reduce_mean(self.score)

        if self.config.log.enable_summary:
            with tf.name_scope("summary"):
                with tf.name_scope("disc_summary"):
                    tf.summary.scalar("loss_discriminator_total", self.loss_discriminator, ["dis"])
                    tf.summary.scalar("loss_dis_real", self.loss_dis_real, ["dis"])
                    tf.summary.scalar("loss_dis_fake", self.loss_dis_fake, ["dis"])
                    if self.config.trainer.loss_method:
                        tf.summary.scalar("loss_dis_fm", self.feature_match, ["dis"])
                with tf.name_scope("gen_summary"):
                    tf.summary.scalar("loss_generator_total", self.gen_loss_total, ["gen"])
                    tf.summary.scalar("loss_gen_adv", self.gen_loss_ce, ["gen"])
                    tf.summary.scalar("loss_gen_con", self.gen_loss_con, ["gen"])
                    tf.summary.scalar("loss_gen_enc", self.gen_loss_enc, ["gen"])
                with tf.name_scope("image_summary"):
                    tf.summary.image("reconstruct", self.img_rec, 3, ["image"])
                    tf.summary.image("input_images", self.image_input, 3, ["image"])
        if self.config.trainer.enable_early_stop:
            with tf.name_scope("validation_summary"):
                tf.summary.scalar("valid", self.rec_error_valid, ["v"])

        self.sum_op_dis = tf.summary.merge_all("dis")
        self.sum_op_gen = tf.summary.merge_all("gen")
        self.sum_op_im = tf.summary.merge_all("image")
        self.sum_op_valid = tf.summary.merge_all("v")

    def generator(self, image_input, getter=None):
        # This generator will take the image from the input dataset, and first it will
        # it will create a latent representation of that image then with the decoder part,
        # it will reconstruct the image.
        with tf.variable_scope("Generator", custom_getter=getter, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("Encoder_1"):
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
                    x_e = tf.nn.leaky_relu(
                        features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                    )
                    # 14 x 14 x 64
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
                        x_e,
                        momentum=self.config.trainer.batch_momentum,
                        epsilon=self.config.trainer.batch_epsilon,
                        training=self.is_training,
                    )
                    x_e = tf.nn.leaky_relu(
                        features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                    )
                    # 7 x 7 x 128
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
                        x_e,
                        momentum=self.config.trainer.batch_momentum,
                        epsilon=self.config.trainer.batch_epsilon,
                        training=self.is_training,
                    )
                    x_e = tf.nn.leaky_relu(
                        features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                    )
                    # 4 x 4 x 256
                x_e = tf.layers.Flatten()(x_e)
                net_name = "Layer_4"
                with tf.variable_scope(net_name):
                    x_e = tf.layers.Dense(
                        units=self.config.trainer.noise_dim,
                        kernel_initializer=self.init_kernel,
                        name="fc",
                    )(x_e)

            noise_gen = x_e

            with tf.variable_scope("Decoder"):
                net = tf.reshape(noise_gen, [-1, 1, 1, self.config.trainer.noise_dim])
                net_name = "layer_1"
                with tf.variable_scope(net_name):
                    net = tf.layers.Conv2DTranspose(
                        filters=256,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="tconv1",
                    )(net)
                    net = tf.layers.batch_normalization(
                        inputs=net,
                        momentum=self.config.trainer.batch_momentum,
                        epsilon=self.config.trainer.batch_epsilon,
                        training=self.is_training,
                        name="tconv1/bn",
                    )
                    net = tf.nn.relu(features=net, name="tconv1/relu")

                net_name = "layer_2"
                with tf.variable_scope(net_name):
                    net = tf.layers.Conv2DTranspose(
                        filters=128,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="tconv2",
                    )(net)
                    net = tf.layers.batch_normalization(
                        inputs=net,
                        momentum=self.config.trainer.batch_momentum,
                        epsilon=self.config.trainer.batch_epsilon,
                        training=self.is_training,
                        name="tconv2/bn",
                    )
                    net = tf.nn.relu(features=net, name="tconv2/relu")

                net_name = "layer_3"
                with tf.variable_scope(net_name):
                    net = tf.layers.Conv2DTranspose(
                        filters=64,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="tconv3",
                    )(net)
                    net = tf.layers.batch_normalization(
                        inputs=net,
                        momentum=self.config.trainer.batch_momentum,
                        epsilon=self.config.trainer.batch_epsilon,
                        training=self.is_training,
                        name="tconv3/bn",
                    )
                    net = tf.nn.relu(features=net, name="tconv3/relu")
                net_name = "layer_4"
                with tf.variable_scope(net_name):
                    net = tf.layers.Conv2DTranspose(
                        filters=32,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="tconv4",
                    )(net)
                    net = tf.layers.batch_normalization(
                        inputs=net,
                        momentum=self.config.trainer.batch_momentum,
                        epsilon=self.config.trainer.batch_epsilon,
                        training=self.is_training,
                        name="tconv4/bn",
                    )
                    net = tf.nn.relu(features=net, name="tconv4/relu")
                net_name = "layer_5"
                with tf.variable_scope(net_name):
                    net = tf.layers.Conv2DTranspose(
                        filters=16,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="tconv5",
                    )(net)
                    net = tf.layers.batch_normalization(
                        inputs=net,
                        momentum=self.config.trainer.batch_momentum,
                        epsilon=self.config.trainer.batch_epsilon,
                        training=self.is_training,
                        name="tconv4/bn",
                    )
                    net = tf.nn.relu(features=net, name="tconv5/relu")

                net_name = "layer_6"
                with tf.variable_scope(net_name):
                    net = tf.layers.Conv2DTranspose(
                        filters=1,
                        kernel_size=5,
                        strides=(1, 1),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="tconv6",
                    )(net)
                    net = tf.nn.tanh(net, name="tconv6/tanh")

            image_rec = net

            # Second Encoder
            with tf.variable_scope("Encoder_2"):
                x_e_2 = tf.reshape(
                    image_rec,
                    [-1, self.config.data_loader.image_size, self.config.data_loader.image_size, 1],
                )
                net_name = "Layer_1"
                with tf.variable_scope(net_name):
                    x_e_2 = tf.layers.Conv2D(
                        filters=64,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="conv",
                    )(x_e_2)
                    x_e_2 = tf.nn.leaky_relu(
                        features=x_e_2, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                    )
                net_name = "Layer_2"
                with tf.variable_scope(net_name):
                    x_e_2 = tf.layers.Conv2D(
                        filters=128,
                        kernel_size=5,
                        padding="same",
                        strides=(2, 2),
                        kernel_initializer=self.init_kernel,
                        name="conv",
                    )(x_e_2)
                    x_e_2 = tf.layers.batch_normalization(
                        x_e_2,
                        momentum=self.config.trainer.batch_momentum,
                        epsilon=self.config.trainer.batch_epsilon,
                        training=self.is_training,
                    )
                    x_e_2 = tf.nn.leaky_relu(
                        features=x_e_2, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                    )
                net_name = "Layer_3"
                with tf.variable_scope(net_name):
                    x_e_2 = tf.layers.Conv2D(
                        filters=256,
                        kernel_size=5,
                        padding="same",
                        strides=(2, 2),
                        kernel_initializer=self.init_kernel,
                        name="conv",
                    )(x_e_2)
                    x_e_2 = tf.layers.batch_normalization(
                        x_e_2,
                        momentum=self.config.trainer.batch_momentum,
                        epsilon=self.config.trainer.batch_epsilon,
                        training=self.is_training,
                    )
                    x_e_2 = tf.nn.leaky_relu(
                        features=x_e_2, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                    )
                x_e_2 = tf.layers.Flatten()(x_e_2)
                net_name = "Layer_4"
                with tf.variable_scope(net_name):
                    x_e_2 = tf.layers.Dense(
                        units=self.config.trainer.noise_dim,
                        kernel_initializer=self.init_kernel,
                        name="fc",
                    )(x_e_2)
            noise_rec = x_e_2
            return noise_gen, image_rec, noise_rec

    def discriminator(self, image, getter=None):
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
                )(image)
                x_d = tf.layers.batch_normalization(
                    inputs=x_d,
                    momentum=self.config.trainer.batch_momentum,
                    epsilon=self.config.trainer.batch_epsilon,
                    training=self.is_training,
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
                    epsilon=self.config.trainer.batch_epsilon,
                    training=self.is_training,
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
                    epsilon=self.config.trainer.batch_epsilon,
                    training=self.is_training,
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
                    epsilon=self.config.trainer.batch_epsilon,
                    training=self.is_training,
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

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.log.max_to_keep)
