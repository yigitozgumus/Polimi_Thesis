import tensorflow as tf

from base.base_model import BaseModel
import utils.alad_utils as sn


class ALAD(BaseModel):
    def __init__(self, config):
        """
        Args:
            config:
        """
        super(ALAD, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):

        # Placeholdersn
        self.init_kernel = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        self.is_training = tf.placeholder(tf.bool)
        self.image_tensor = tf.placeholder(
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
        # Building the Graph
        with tf.variable_scope("ALAD"):
            # Generated noise from the encoder
            with tf.variable_scope("Encoder_Model"):
                self.z_gen = self.encoder(
                    self.image_tensor, do_spectral_norm=self.config.trainer.do_spectral_norm
                )
            # Generated image and reconstructed image from the Generator
            with tf.variable_scope("Generator_Model"):
                self.img_gen = self.generator(self.noise_tensor) + self.fake_noise
                self.rec_img = self.generator(self.z_gen)

            # Reconstructed image of generated image from the encoder
            with tf.variable_scope("Encoder_Model"):
                self.rec_z = self.encoder(self.img_gen, do_spectral_norm=self.config.spectral_norm)

            # Discriminator results of (G(z),z) and (x, E(x))
            with tf.variable_scope("Discriminator_Model_XZ"):
                l_generator, inter_layer_rct_xz = self.discriminator_xz(
                    self.img_gen, self.noise_tensor, do_spectral_norm=self.config.spectral_norm
                )
                l_encoder, inter_layer_inp_xz = self.discriminator_xz(
                    self.image_tensor + self.real_noise,
                    self.z_gen,
                    do_spectral_norm=self.config.do_spectral_norm,
                )

            # Discrimeinator results of (x, x) and (x, G(E(x))
            with tf.variable_scope("Discriminator_Model_XX"):
                x_logit_real, inter_layer_inp_xx = self.discriminator_xx(
                    self.image_tensor + self.real_noise,
                    self.image_tensor + self.real_noise,
                    do_spectral_norm=self.config.spectral_norm,
                )
                x_logit_fake, inter_layer_rct_xx = self.discriminator_xx(
                    self.image_tensor + self.real_noise,
                    self.rec_img,
                    do_spectral_norm=self.config.spectral_norm,
                )
            # Discriminator results of (z, z) and (z, E(G(z))
            with tf.variable_scope("Discriminator_Model_ZZ"):
                z_logit_real, _ = self.discriminator_zz(
                    self.noise_tensor, self.noise_tensor, do_spectral_norm=self.config.spectral_norm
                )
                z_logit_fake, _ = self.discriminator_zz(
                    self.noise_tensor, self.rec_z, do_spectral_norm=self.config.spectral_norm
                )
        ########################################################################
        # LOSS FUNCTIONS
        ########################################################################
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
                logits=x_logit_real, labels=tf.ones_like(x_logit_real)
            )
            x_fake_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_logit_fake, labels=tf.zeros_like(x_logit_fake)
            )
            self.dis_loss_xx = tf.reduce_mean(x_real_dis + x_fake_dis)
            # discriminator zz
            z_real_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=z_logit_real, labels=tf.ones_like(z_logit_real)
            )
            z_fake_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=z_logit_fake, labels=tf.zeros_like(z_logit_fake)
            )
            self.dis_loss_zz = tf.reduce_mean(z_real_dis + z_fake_dis)
            # Compute the whole discriminator loss
            self.loss_discriminator = (
                self.dis_loss_xz + self.dis_loss_xx + self.dis_loss_zz
                if self.config.trainer.allow_zz
                else self.dis_loss_xz + self.dis_loss_xx
            )
            # generator and encoder
            if self.config.trainer.flip_labels:
                labels_gen = tf.zeros_like(l_generator)
                labels_enc = tf.ones_like(l_encoder)
            else:
                labels_gen = tf.ones_like(l_generator)
                labels_enc = tf.zeros_like(l_encoder)

            gen_loss_xz = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_gen, logits=l_generator)
            )
            enc_loss_xz = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_enc, logits=l_encoder)
            )

            x_real_gen = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_logit_real, labels=self.generated_labels
            )
            x_fake_gen = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x_logit_fake, labels=self.true_labels
            )
            z_real_gen = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=z_logit_real, labels=self.generated_labels
            )
            z_fake_gen = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=z_logit_fake, labels=self.true_labels
            )

            cost_x = tf.reduce_mean(x_real_gen + x_fake_gen)
            cost_z = tf.reduce_mean(z_real_gen + z_fake_gen)

            cycle_consistency_loss = cost_x + cost_z if self.config.trainer.allow_zz else cost_x
            self.loss_generator = gen_loss_xz + cycle_consistency_loss
            self.loss_encoder = enc_loss_xz + cycle_consistency_loss

        ########################################################################
        # OPTIMIZATION
        ########################################################################
        with tf.name_scope("Optimizers"):

            # control op dependencies for batch norm and trainable variables
            all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.dxzvars = [
                v for v in all_variables if v.name.startswith("ALAD/Discriminator_Model_XZ")
            ]
            self.dxxvars = [
                v for v in all_variables if v.name.startswith("ALAD/Discriminator_Model_XX")
            ]
            self.dzzvars = [
                v for v in all_variables if v.name.startswith("ALAD/Discriminator_Model_ZZ")
            ]
            self.gvars = [v for v in all_variables if v.name.startswith("ALAD/Generator_Model")]
            self.evars = [v for v in all_variables if v.name.startswith("ALAD/Encoder_Model")]

            self.update_ops_gen = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="ALAD/Generator_Model"
            )
            self.update_ops_enc = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="ALAD/Encoder_Model"
            )
            self.update_ops_dis_xz = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="ALAD/Discriminator_Model_XZ"
            )
            self.update_ops_dis_xx = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="ALAD/Discriminator_Model_XX"
            )
            self.update_ops_dis_zz = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="ALAD/Discriminator_Model_ZZ"
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
            self.enc_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.trainer.generator_l_rate,
                beta1=self.config.trainer.optimizer_adam_beta1,
                beta2=self.config.trainer.optimizer_adam_beta2,
            )

            with tf.control_dependencies(self.update_ops_gen):
                self.gen_op = self.gen_optimizer.minimize(
                    self.loss_generator, global_step=self.global_step_tensor, var_list=self.gvars
                )
            with tf.control_dependencies(self.update_ops_enc):
                self.enc_op = self.enc_optimizer.minimize(self.loss_encoder, var_list=self.evars)

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
            self.train_enc_op, self.enc_ema = train_op_with_ema_dependency(self.evars, self.enc_op)

            self.train_dis_op_xz, self.xz_ema = train_op_with_ema_dependency(
                self.dxzvars, self.dis_op_xz
            )
            self.train_dis_op_xx, self.xx_ema = train_op_with_ema_dependency(
                self.dxxvars, self.dis_op_xx
            )
            self.train_dis_op_zz, self.zz_ema = train_op_with_ema_dependency(
                self.dzzvars, self.dis_op_zz
            )

        with tf.variable_scope("ALAD"):
            with tf.variable_scope("Encoder_Model"):
                self.z_gen_ema = self.encoder(
                    self.image_tensor,
                    getter=sn.get_getter(self.enc_ema),
                    do_spectral_norm=self.config.trainer.spectral_norm,
                )

            with tf.variable_scope("Generator_Model"):
                self.rec_x_ema = self.generator(self.z_gen_ema, getter=sn.get_getter(self.gen_ema))
                self.x_gen_ema = self.generator(
                    self.noise_tensor, getter=sn.get_getter(self.gen_ema)
                )
            with tf.variable_scope("Discriminator_Model_XX"):
                l_encoder_emaxx, inter_layer_inp_emaxx = self.discriminator_xx(
                    self.image_tensor,
                    self.image_tensor,
                    getter=sn.get_getter(self.xx_ema),
                    do_spectral_norm=self.config.trainer.spectral_norm,
                )
                l_generator_emaxx, inter_layer_rct_emaxx = self.discriminator_xx(
                    self.image_tensor,
                    self.rec_x_ema,
                    getter=sn.get_getter(self.xx_ema),
                    do_spectral_norm=self.config.trainer.spectral_norm,
                )

        with tf.name_scope("Testing"):
            with tf.variable_scope("Scores"):
                score_ch = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(l_generator_emaxx), logits=l_generator_emaxx
                )
                self.score_ch = tf.squeeze(score_ch)

                rec = self.image_tensor - self.rec_x_ema
                rec = tf.layers.Flatten()(rec)
                score_l1 = tf.norm(rec, ord=1, axis=1, keepdims=False, name="d_loss")
                self.score_l1 = tf.squeeze(score_l1)

                rec = self.image_tensor - self.rec_x_ema
                rec = tf.layers.Flatten()(rec)
                score_l2 = tf.norm(rec, ord=2, axis=1, keepdims=False, name="d_loss")
                self.score_l2 = tf.squeeze(score_l2)

                inter_layer_inp, inter_layer_rct = (inter_layer_inp_emaxx, inter_layer_rct_emaxx)
                fm = inter_layer_inp - inter_layer_rct
                fm = tf.layers.Flatten()(fm)
                score_fm = tf.norm(
                    fm, ord=self.config.trainer.degree, axis=1, keepdims=False, name="d_loss"
                )
                self.score_fm = tf.squeeze(score_fm)

        if self.config.trainer.enable_early_stop:
            self.rec_error_valid = tf.reduce_mean(score_fm)
        ########################################################################
        # TENSORBOARD
        ########################################################################
        if self.config.log.enable_summary:

            with tf.name_scope("train_summary"):

                with tf.name_scope("dis_summary"):
                    tf.summary.scalar("loss_discriminator", self.loss_discriminator, ["dis"])
                    tf.summary.scalar("loss_dis_encoder", loss_dis_enc, ["dis"])
                    tf.summary.scalar("loss_dis_gen", loss_dis_gen, ["dis"])
                    tf.summary.scalar("loss_dis_xz", self.dis_loss_xz, ["dis"])
                    tf.summary.scalar("loss_dis_xx", self.dis_loss_xx, ["dis"])
                    if self.config.trainer.allow_zz:
                        tf.summary.scalar("loss_dis_zz", self.dis_loss_zz, ["dis"])

                with tf.name_scope("gen_summary"):

                    tf.summary.scalar("loss_generator", self.loss_generator, ["gen"])
                    tf.summary.scalar("loss_encoder", self.loss_encoder, ["gen"])
                    tf.summary.scalar("loss_encgen_dxx", cost_x, ["gen"])
                    if self.config.trainer.allow_zz:
                        tf.summary.scalar("loss_encgen_dzz", cost_z, ["gen"])

                with tf.name_scope("img_summary"):
                    heatmap_pl_latent = tf.placeholder(
                        tf.float32, shape=(1, 480, 640, 3), name="heatmap_pl_latent"
                    )
                    self.sum_op_latent = tf.summary.image("heatmap_latent", heatmap_pl_latent)

                with tf.name_scope("image_summary"):
                    tf.summary.image("reconstruct", self.rec_img, 3, ["image"])
                    tf.summary.image("input_images", self.image_tensor, 3, ["image"])

        if self.config.trainer.enable_early_stop:
            with tf.name_scope("validation_summary"):
                tf.summary.scalar("valid", self.rec_error_valid, ["v"])

        self.sum_op_dis = tf.summary.merge_all("dis")
        self.sum_op_gen = tf.summary.merge_all("gen")
        self.sum_op = tf.summary.merge([self.sum_op_dis, self.sum_op_gen])
        self.sum_op_im = tf.summary.merge_all("image")
        self.sum_op_valid = tf.summary.merge_all("v")

    def encoder(self, img_tensor, getter=None, do_spectral_norm=True):

        """ Encoder architecture in tensorflow
        Maps the image data to the latent space
        Args:
            img_tensor: input data for the encoder
            getter: for exponential moving average during inference
            reuse: sharing variables or not
            do_spectral_norm:
        """
        # Change the layer type if do_spectral_norm is true
        layers = sn if do_spectral_norm else tf.layers
        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE, custom_getter=getter):
            img_tensor = tf.reshape(
                img_tensor,
                [-1, self.config.data_loader.image_size, self.config.data_loader.image_size, 1],
            )
            net_name = "layer_1"
            with tf.variable_scope(net_name):
                net = layers.conv2d(
                    img_tensor,
                    filters=32,
                    kernel_size=4,
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )
                net = tf.layers.batch_normalization(
                    inputs=net,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="bn",
                )
                net = tf.nn.leaky_relu(
                    features=net, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )

            net_name = "layer_2"
            with tf.variable_scope(net_name):
                net = layers.conv2d(
                    net,
                    filters=64,
                    kernel_size=4,
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )
                net = tf.layers.batch_normalization(
                    inputs=net,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="bn",
                )
                net = tf.nn.leaky_relu(
                    features=net, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )

            net_name = "layer_3"
            with tf.variable_scope(net_name):
                net = layers.conv2d(
                    net,
                    filters=128,
                    kernel_size=4,
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )
                net = tf.layers.batch_normalization(
                    inputs=net,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="bn",
                )
                net = tf.nn.leaky_relu(
                    features=net, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
            net_name = "layer_4"
            with tf.variable_scope(net_name):
                net = layers.conv2d(
                    net,
                    filters=256,
                    kernel_size=4,
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )
                net = tf.layers.batch_normalization(
                    inputs=net,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="bn",
                )
                net = tf.nn.leaky_relu(
                    features=net, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )

            net_name = "layer_5"
            with tf.variable_scope(net_name):
                net = layers.conv2d(
                    net,
                    filters=self.config.trainer.noise_dim,
                    kernel_size=4,
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )
                net = tf.layers.batch_normalization(
                    inputs=net,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="bn",
                )
                net = tf.nn.leaky_relu(
                    features=net, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
                net = tf.squeeze(net, [1, 2])
        return net

    def generator(self, noise_tensor, getter=None):
        """ Generator architecture in tensorflow

        Generates the data from the latent space
        Args:
            noise_tensor: input variable in the latent space
            getter: for exponential moving average during inference
            reuse: sharing variables or not
        """
        with tf.variable_scope("Generator", reuse=tf.AUTO_REUSE, custom_getter=getter):
            net = tf.reshape(noise_tensor, [-1, 1, 1, self.config.trainer.noise_dim])
            net_name = "layer_1"
            with tf.variable_scope(net_name):
                net = tf.layers.Dense(
                    units=4 * 4 * 512, kernel_initializer=self.init_kernel, name="fc"
                )(net)
                net = tf.layers.batch_normalization(
                    inputs=net,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="tconv1/bn",
                )
                net = tf.nn.leaky_relu(
                    features=net, alpha=self.config.trainer.leakyReLU_alpha, name="tconv1/relu"
                )
                net = tf.reshape(net, shape=[-1, 4, 4, 512])
            net_name = "layer_2"
            with tf.variable_scope(net_name):
                net = tf.layers.Conv2DTranspose(
                    filters=512,
                    kernel_size=4,
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="tconv2",
                )(net)
                net = tf.layers.batch_normalization(
                    inputs=net,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="tconv2/bn",
                )
                net = tf.nn.leaky_relu(
                    features=net, alpha=self.config.trainer.leakyReLU_alpha, name="tconv2/relu"
                )

            net_name = "layer_3"
            with tf.variable_scope(net_name):
                net = tf.layers.Conv2DTranspose(
                    filters=256,
                    kernel_size=4,
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="tconv3",
                )(net)
                net = tf.layers.batch_normalization(
                    inputs=net,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="tconv3/bn",
                )
                net = tf.nn.leaky_relu(
                    features=net, alpha=self.config.trainer.leakyReLU_alpha, name="tconv3/relu"
                )

            net_name = "layer_4"
            with tf.variable_scope(net_name):
                net = tf.layers.Conv2DTranspose(
                    filters=128,
                    kernel_size=4,
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="tconv4",
                )(net)
                net = tf.layers.batch_normalization(
                    inputs=net,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="tconv4/bn",
                )
                net = tf.nn.leaky_relu(
                    features=net, alpha=self.config.trainer.leakyReLU_alpha, name="tconv4/relu"
                )
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

        return net

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
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
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
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
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
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
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
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
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
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
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
                    strides=(1, 1),
                    padding="same",
                    kernel_initializer=self.init_kernel,
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
                    strides=(1, 1),
                    padding="same",
                    kernel_initializer=self.init_kernel,
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
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=self.init_kernel,
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
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=self.init_kernel,
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
                net = tf.layers.dense(net, units=1, kernel_initializer=self.init_kernel, name="fc")
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
                y = layers.dense(y, units=64, kernel_initializer=self.init_kernel, name="fc")
                y = tf.nn.leaky_relu(features=y, alpha=self.config.trainer.leakyReLU_alpha)
                y = tf.layers.dropout(
                    y,
                    rate=self.config.trainer.dropout_rate,
                    training=self.is_training,
                    name="dropout",
                )

            net_name = "y_layer_2"
            with tf.variable_scope(net_name):
                y = layers.dense(y, units=32, kernel_initializer=self.init_kernel, name="fc")
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
                y = layers.dense(y, units=1, kernel_initializer=self.init_kernel, name="fc")
                logits = tf.squeeze(y)

        return logits, intermediate_layer

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.log.max_to_keep)
