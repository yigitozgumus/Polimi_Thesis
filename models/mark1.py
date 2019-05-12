import tensorflow as tf

from base.base_model import BaseModel
from utils.alad_utils import get_getter
import utils.alad_utils as sn


class Mark1(BaseModel):
    def __init__(self, config):
        """
        Args:
            config:
        """
        super(Mark1, self).__init__(config)
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

        self.logger.info("Building training graph...")

        with tf.variable_scope("Mark1"):
            with tf.variable_scope("Generator_Model"):
                self.noise_gen, self.img_rec, self.noise_rec = self.generator(self.image_input)
            # Discriminator results of (G(z),z) and (x, E(x))
            with tf.variable_scope("Discriminator_Model_XZ"):
                l_generator, inter_layer_rct_xz = self.discriminator_xz(
                    self.img_rec, self.noise_gen, do_spectral_norm=self.config.spectral_norm
                )
                l_encoder, inter_layer_inp_xz = self.discriminator_xz(
                    self.image_input, self.noise_gen, do_spectral_norm=self.config.do_spectral_norm
                )
            # Discrimeinator results of (x, x) and (x, G(E(x))
            with tf.variable_scope("Discriminator_Model_XX"):
                x_logit_real, inter_layer_inp_xx = self.discriminator_xx(
                    self.image_input, self.image_input, do_spectral_norm=self.config.spectral_norm
                )
                x_logit_fake, inter_layer_rct_xx = self.discriminator_xx(
                    self.image_input, self.img_rec, do_spectral_norm=self.config.spectral_norm
                )
            # Discriminator results of (z, z) and (z, E(G(z))
            with tf.variable_scope("Discriminator_Model_ZZ"):
                z_logit_real, _ = self.discriminator_zz(
                    self.noise_gen, self.noise_gen, do_spectral_norm=self.config.spectral_norm
                )
                z_logit_fake, _ = self.discriminator_zz(
                    self.noise_gen, self.noise_rec, do_spectral_norm=self.config.spectral_norm
                )

        with tf.name_scope("Loss_Functions"):
            # discriminator xz

            # Discriminator should classify encoder pair as real
            loss_dis_enc = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.true_labels, logits=l_encoder)
            )
            # Discriminator should classify generator pair as fake
            loss_dis_gen = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.generated_labels, logits=l_generator
                )
            )
            self.dis_loss_xz = loss_dis_gen + loss_dis_enc

            # discriminator xx
            x_real_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_logit_real, labels=self.true_labels
            )
            x_fake_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_logit_fake, labels=self.generated_labels
            )
            self.dis_loss_xx = tf.reduce_mean(x_real_dis + x_fake_dis)
            # discriminator zz
            z_real_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=z_logit_real, labels=self.true_labels
            )
            z_fake_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=z_logit_fake, labels=self.generated_labels
            )
            self.dis_loss_zz = tf.reduce_mean(z_real_dis + z_fake_dis)
            # Feature matching part
            fm = inter_layer_inp_xx - inter_layer_rct_xx
            fm = tf.layers.Flatten()(fm)
            self.feature_match = tf.reduce_mean(tf.norm(fm, ord=2, axis=1, keepdims=False))
            # Compute the whole discriminator loss
            self.loss_discriminator = (
                self.dis_loss_xz + self.dis_loss_xx + self.dis_loss_zz
                if self.config.trainer.allow_zz
                else self.dis_loss_xz + self.dis_loss_xx + self.feature_match
                if self.config.trainer.loss_method == "fm"
                else self.dis_loss_xz + self.dis_loss_xx
            )
            # Generator
            # Adversarial Loss
            if self.config.trainer.flip_labels:
                labels_gen = tf.zeros_like(l_generator)
                labels_enc = tf.ones_like(l_encoder)
            else:
                labels_gen = tf.ones_like(l_generator)
                labels_enc = tf.zeros_like(l_encoder)
            self.gen_loss_enc = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_enc, logits=l_encoder)
            )
            self.gen_loss_gen = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_gen, logits=l_generator)
            )
            # Contextual Loss
            l1_norm = self.image_input - self.img_rec
            l1_norm = tf.layers.Flatten()(l1_norm)
            self.gen_loss_con = tf.reduce_mean(tf.norm(l1_norm, ord=1, axis=1, keepdims=False))
            # Encoder Loss
            l2_norm = self.noise_gen - self.noise_rec
            l2_norm = tf.layers.Flatten()(l2_norm)
            self.gen_loss_enc = tf.reduce_mean(tf.norm(l2_norm, ord=2, axis=1, keepdims=False))

            self.gen_loss_ce = self.gen_loss_enc + self.gen_loss_gen
            self.gen_loss_total = (
                self.config.trainer.weight_adv * self.gen_loss_ce
                + self.config.trainer.weight_cont * self.gen_loss_con
                + self.config.trainer.weight_enc * self.gen_loss_enc
            )

        with tf.name_scope("Optimizers"):
            # Collect all the variables
            all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # Generator Network Variables
            self.gvars = [v for v in all_variables if v.name.startswith("Mark1/Generator_Model")]
            self.dxzvars = [
                v for v in all_variables if v.name.startswith("Mark1/Discriminator_Model_XZ")
            ]
            self.dxxvars = [
                v for v in all_variables if v.name.startswith("Mark1/Discriminator_Model_XX")
            ]
            self.dzzvars = [
                v for v in all_variables if v.name.startswith("Mark1/Discriminator_Model_ZZ")
            ]
            # Create Training Operations
            # Generator Network Operations
            self.update_ops_gen = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="Mark1/Generator_Model"
            )
            self.update_ops_enc = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="Mark1/Encoder_Model"
            )
            self.update_ops_dis_xz = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="Mark1/Discriminator_Model_XZ"
            )
            self.update_ops_dis_xx = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="Mark1/Discriminator_Model_XX"
            )
            self.update_ops_dis_zz = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="Mark1/Discriminator_Model_ZZ"
            )
            self.disc_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.trainer.discriminator_l_rate,
                beta1=self.config.trainer.optimizer_adam_beta1,
                beta2=self.config.trainer.optimizer_adam_beta2,
            )
            self.gen_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.trainer.generator_l_rate,
                beta1=self.config.trainer.optimizer_adam_beta1,
                beta2=self.config.trainer.optimizer_adam_beta2,
            )
            # Initialization of Optimizers
            with tf.control_dependencies(self.update_ops_gen):
                self.gen_op = self.gen_optimizer.minimize(
                    self.gen_loss_total, global_step=self.global_step_tensor, var_list=self.gvars
                )

            with tf.control_dependencies(self.update_ops_dis_xz):
                self.dis_op_xz = self.disc_optimizer.minimize(
                    self.dis_loss_xz, var_list=self.dxzvars
                )

            with tf.control_dependencies(self.update_ops_dis_xx):
                self.dis_op_xx = self.disc_optimizer.minimize(
                    self.dis_loss_xx, var_list=self.dxxvars
                )

            with tf.control_dependencies(self.update_ops_dis_zz):
                self.dis_op_zz = self.disc_optimizer.minimize(
                    self.dis_loss_zz, var_list=self.dzzvars
                )

            # Exponential Moving Average for inference
            def train_op_with_ema_dependency(vars, op):
                ema = tf.train.ExponentialMovingAverage(decay=self.config.trainer.ema_decay)
                maintain_averages_op = ema.apply(vars)
                with tf.control_dependencies([op]):
                    train_op = tf.group(maintain_averages_op)
                return train_op, ema

            self.train_gen_op, self.gen_ema = train_op_with_ema_dependency(self.gvars, self.gen_op)

            self.train_dis_op_xz, self.xz_ema = train_op_with_ema_dependency(
                self.dxzvars, self.dis_op_xz
            )
            self.train_dis_op_xx, self.xx_ema = train_op_with_ema_dependency(
                self.dxxvars, self.dis_op_xx
            )
            self.train_dis_op_zz, self.zz_ema = train_op_with_ema_dependency(
                self.dzzvars, self.dis_op_zz
            )

        self.logger.info("Building Testing Graph...")
        with tf.variable_scope("Mark1"):
            with tf.variable_scope("Generator_Model"):
                self.noise_gen_ema, self.img_rec_ema, self.noise_rec_ema = self.generator(
                    self.image_input, getter=get_getter(self.gen_ema)
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
                    tf.summary.scalar("loss_discriminator", self.loss_discriminator, ["dis"])
                    tf.summary.scalar("loss_dis_encoder", loss_dis_enc, ["dis"])
                    tf.summary.scalar("loss_dis_gen", loss_dis_gen, ["dis"])
                    tf.summary.scalar("loss_dis_xz", self.dis_loss_xz, ["dis"])
                    tf.summary.scalar("loss_dis_xx", self.dis_loss_xx, ["dis"])
                    if self.config.trainer.allow_zz:
                        tf.summary.scalar("loss_dis_zz", self.dis_loss_zz, ["dis"])
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
                        filters=512,
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
                        filters=256,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="valid",
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
                        filters=128,
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
                        filters=64,
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
                    net = tf.nn.relu(features=net, name="tconv3/relu")

                net_name = "layer_5"
                with tf.variable_scope(net_name):
                    net = tf.layers.Conv2DTranspose(
                        filters=1,
                        kernel_size=5,
                        strides=(1, 1),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="tconv5",
                    )(net)
                    net = tf.nn.tanh(net, name="tconv5/tanh")

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

    def discriminator_xz(self, img_tensor, noise_tensor, getter=None, do_spectral_norm=True):
        """ Discriminator architecture in tensorflow

        Discriminates between pairs (E(x), x) and (z, G(z))
        Args:
            img_tensor:
            noise_tensor:
            getter: for exponential moving average during inference
            reuse: sharing variables or not
            do_spectral_norm:
        """
        layers = sn if do_spectral_norm else tf.layers
        with tf.variable_scope("Discriminator_xz", reuse=tf.AUTO_REUSE, custom_getter=getter):
            net_name = "x_layer_1"
            with tf.variable_scope(net_name):
                x = layers.conv2d(
                    img_tensor,
                    filters=128,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                    name="conv1",
                )
                x = tf.nn.leaky_relu(
                    features=x, alpha=self.config.trainer.leakyReLU_alpha, name="conv1/leaky_relu"
                )

            net_name = "x_layer_2"
            with tf.variable_scope(net_name):
                x = layers.conv2d(
                    x,
                    filters=256,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                    name="conv2",
                )
                x = tf.layers.batch_normalization(
                    inputs=x,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="tconv2/bn",
                )
                x = tf.nn.leaky_relu(
                    features=x, alpha=self.config.trainer.leakyReLU_alpha, name="conv2/leaky_relu"
                )
            net_name = "x_layer_3"
            with tf.variable_scope(net_name):
                x = layers.conv2d(
                    x,
                    filters=512,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                    name="conv2",
                )
                x = tf.layers.batch_normalization(
                    inputs=x,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="tconv3/bn",
                )
                x = tf.nn.leaky_relu(
                    features=x, alpha=self.config.trainer.leakyReLU_alpha, name="conv3/leaky_relu"
                )

            x = tf.reshape(x, [-1, 1, 1, 512 * 4 * 4])
            # Reshape the noise
            z = tf.reshape(noise_tensor, [-1, 1, 1, self.config.trainer.noise_dim])

            net_name = "z_layer_1"
            with tf.variable_scope(net_name):
                z = layers.conv2d(
                    z,
                    filters=512,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                    name="conv",
                )
                z = tf.nn.leaky_relu(
                    features=z, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
                z = tf.layers.dropout(
                    z,
                    rate=self.config.trainer.dropout_rate,
                    training=self.is_training,
                    name="dropout",
                )

            net_name = "z_layer_2"
            with tf.variable_scope(net_name):
                z = layers.conv2d(
                    z,
                    filters=512,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                    name="conv",
                )
                z = tf.nn.leaky_relu(
                    features=z, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
                z = tf.layers.dropout(
                    z,
                    rate=self.config.trainer.dropout_rate,
                    training=self.is_training,
                    name="dropout",
                )
            y = tf.concat([x, z], axis=-1)
            net_name = "y_layer_1"
            with tf.variable_scope(net_name):
                y = layers.conv2d(
                    y,
                    filters=1024,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                    name="conv",
                )
                y = tf.nn.leaky_relu(
                    features=y, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
                y = tf.layers.dropout(
                    y,
                    rate=self.config.trainer.dropout_rate,
                    training=self.is_training,
                    name="dropout",
                )

            intermediate_layer = y

            net_name = "y_layer_2"
            with tf.variable_scope(net_name):
                y = layers.conv2d(
                    y,
                    filters=1024,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                    name="conv",
                )
                y = tf.nn.leaky_relu(
                    features=y, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
                y = tf.layers.dropout(
                    y,
                    rate=self.config.trainer.dropout_rate,
                    training=self.is_training,
                    name="dropout",
                )

            logits = tf.squeeze(y)

            return logits, intermediate_layer

    def discriminator_xx(self, img_tensor, recreated_img, getter=None, do_spectral_norm=True):
        """ Discriminator architecture in tensorflow

        Discriminates between (x, x) and (x, rec_x)
        Args:
            img_tensor:
            recreated_img:
            getter: for exponential moving average during inference
            reuse: sharing variables or not
            do_spectral_norm:
        """
        layers = sn if do_spectral_norm else tf.layers
        with tf.variable_scope("Discriminator_xx", reuse=tf.AUTO_REUSE, custom_getter=getter):
            net = tf.concat([img_tensor, recreated_img], axis=1)
            net_name = "layer_1"
            with tf.variable_scope(net_name):
                net = layers.conv2d(
                    net,
                    filters=64,
                    kernel_size=5,
                    strides=2,
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                    name="conv1",
                )
                net = tf.nn.leaky_relu(
                    features=net, alpha=self.config.trainer.leakyReLU_alpha, name="conv2/leaky_relu"
                )
                net = tf.layers.dropout(
                    net,
                    rate=self.config.trainer.dropout_rate,
                    training=self.is_training,
                    name="dropout",
                )
            with tf.variable_scope(net_name, reuse=True):
                weights = tf.get_variable("conv1/kernel")

            net_name = "layer_2"
            with tf.variable_scope(net_name):
                net = layers.conv2d(
                    net,
                    filters=128,
                    kernel_size=5,
                    strides=2,
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                    name="conv2",
                )
                net = tf.nn.leaky_relu(
                    features=net, alpha=self.config.trainer.leakyReLU_alpha, name="conv2/leaky_relu"
                )
                net = tf.layers.dropout(
                    net,
                    rate=self.config.trainer.dropout_rate,
                    training=self.is_training,
                    name="dropout",
                )
            net = tf.layers.Flatten()(net)

            intermediate_layer = net

            net_name = "layer_3"
            with tf.variable_scope(net_name):
                net = tf.layers.dense(
                    net,
                    units=1,
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                    name="fc",
                )
                logits = tf.squeeze(net)

        return logits, intermediate_layer

    def discriminator_zz(self, noise_tensor, recreated_noise, getter=None, do_spectral_norm=True):
        """ Discriminator architecture in tensorflow

        Discriminates between (z, z) and (z, rec_z)
        Args:
            noise_tensor:
            recreated_noise:
            getter: for exponential moving average during inference
            reuse: sharing variables or not
            do_spectral_norm:
        """
        layers = sn if do_spectral_norm else tf.layers

        with tf.variable_scope("Discriminator_zz", reuse=tf.AUTO_REUSE, custom_getter=getter):
            y = tf.concat([noise_tensor, recreated_noise], axis=-1)

            net_name = "y_layer_1"
            with tf.variable_scope(net_name):
                y = layers.dense(
                    y,
                    units=64,
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                    name="fc",
                )
                y = tf.nn.leaky_relu(features=y, alpha=self.config.trainer.leakyReLU_alpha)
                y = tf.layers.dropout(
                    y,
                    rate=self.config.trainer.dropout_rate,
                    training=self.is_training,
                    name="dropout",
                )

            net_name = "y_layer_2"
            with tf.variable_scope(net_name):
                y = layers.dense(
                    y,
                    units=32,
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                    name="fc",
                )
                y = tf.nn.leaky_relu(features=y, alpha=self.config.trainer.leakyReLU_alpha)
                y = tf.layers.dropout(
                    y,
                    rate=self.config.trainer.dropout_rate,
                    training=self.is_training,
                    name="dropout",
                )

            intermediate_layer = y

            net_name = "y_layer_3"
            with tf.variable_scope(net_name):
                y = layers.dense(
                    y,
                    units=1,
                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                    name="fc",
                )
                logits = tf.squeeze(y)

        return logits, intermediate_layer

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.log.max_to_keep)
