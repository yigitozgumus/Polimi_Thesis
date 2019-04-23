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
        self.init_kernel = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        self.true_labels = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="true_labels")
        self.generated_labels = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="gen_labels")
        #######################################################################
        # GRAPH
        ########################################################################
        self.logger.info("Building Training Graph")

        with tf.variable_scope("Skip_GANomaly"):
            with tf.variable_scope("Generator_Model"):
                self.img_rec = self.generator(self.image_input)
            with tf.variable_scope("Discriminator_Model"):
                self.disc_real, self.inter_layer_real = self.discriminator(self.image_input)
                self.disc_fake, self.inter_layer_fake = self.discriminator(self.img_rec)
        ########################################################################
        # METRICS
        ########################################################################
        with tf.variable_scope("Loss_Functions"):
            with tf.variable_scope("Discriminator_Loss"):
                # According to the paper we invert the values for the normal/fake.
                # So normal images should be labeled
                # as zeros
                self.loss_dis_real = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.true_labels, logits=self.disc_real
                    )
                )
                self.loss_dis_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.generated_labels, logits=self.disc_fake
                    )
                )
                # Adversarial Loss Part for Discriminator
                self.loss_discriminator = self.loss_dis_real + self.loss_dis_fake

            with tf.variable_scope("Generator_Loss"):
                # Adversarial Loss
                if self.config.trainer.flip_labels:
                    labels = tf.zeros_like(self.disc_fake)
                else:
                    labels = tf.ones_like(self.disc_fake)
                self.gen_adv_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_fake, labels=labels)
                )
                # Contextual Loss
                context_layers = self.image_input - self.img_rec
                self.contextual_loss = tf.reduce_mean(
                    tf.norm(context_layers, ord=1, axis=1, keepdims=False, name="Contextual_Loss")
                )
                # Latent Loss
                layer_diff = self.inter_layer_real - self.inter_layer_fake
                self.latent_loss = tf.reduce_mean(
                    tf.norm(layer_diff, ord=2, axis=1, keepdims=False, name="Latent_Loss")
                )
                self.gen_loss_total = (
                    self.config.trainer.weight_adv * self.gen_adv_loss
                    + self.config.trainer.weight_cont * self.contextual_loss
                    + self.config.trainer.weight_lat * self.latent_loss
                )

        ########################################################################
        # OPTIMIZATION
        ########################################################################
        # Build the Optimizers
        with tf.name_scope("Optimization"):
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
                v for v in all_variables if v.name.startswith("Skip_GANomaly/Generator_Model")
            ]
            # Discriminator Network Variables
            self.discriminator_vars = [
                v for v in all_variables if v.name.startswith("Skip_GANomaly/Discriminator_Model")
            ]
            # Generator Network Operations
            self.gen_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="Skip_GANomaly/Generator_Model"
            )
            # Discriminator Network Operations
            self.disc_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="Skip_GANomaly/Discriminator_Model"
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

        ########################################################################
        # TESTING
        ########################################################################
        self.logger.info("Building Testing Graph...")

        with tf.variable_scope("Skip_GANomaly"):
            with tf.variable_scope("Generator_Model"):
                self.img_rec_ema = self.generator(self.image_input, getter=get_getter(self.gen_ema))
            with tf.variable_scope("Discriminator_Model"):
                self.disc_real_ema, self.inter_layer_real_ema = self.discriminator(
                    self.image_input, getter=get_getter(self.dis_ema)
                )
                self.disc_fake_ema, self.inter_layer_fake_ema = self.discriminator(
                    self.img_rec, getter=get_getter(self.dis_ema)
                )

        with tf.name_scope("Testing"):
            with tf.variable_scope("Reconstruction_Loss"):
                # Contextual Loss
                context_layers = self.image_input - self.img_rec_ema
                context_layers = tf.layers.Flatten()(context_layers)
                contextual_loss_ema = tf.norm(
                    context_layers, ord=1, axis=1, keepdims=False, name="Contextual_Loss"
                )
                self.contextual_loss_ema = tf.squeeze(contextual_loss_ema)

            with tf.variable_scope("Latent_Loss"):
                # Latent Loss
                layer_diff = self.inter_layer_real_ema - self.inter_layer_fake_ema
                layer_diff = tf.layers.Flatten()(layer_diff)
                latent_loss_ema = tf.norm(
                    layer_diff, ord=2, axis=1, keepdims=False, name="Latent_Loss"
                )
                self.latent_loss_ema = tf.squeeze(latent_loss_ema)

            self.anomaly_score = self.config.trainer.weight * self.contextual_loss_ema + (
                1 - self.config.trainer.weight * self.latent_loss_ema
            )
        if self.config.trainer.enable_early_stop:
            self.rec_error_valid = tf.reduce_mean(self.latent_loss_ema)

        ########################################################################
        # TENSORBOARD
        ########################################################################
        if self.config.log.enable_summary:
            with tf.name_scope("summary"):
                with tf.name_scope("disc_summary"):
                    tf.summary.scalar("loss_discriminator_total", self.loss_discriminator, ["dis"])
                    tf.summary.scalar("loss_dis_real", self.loss_dis_real, ["dis"])
                    tf.summary.scalar("loss_dis_fake", self.loss_dis_fake, ["dis"])
                with tf.name_scope("gen_summary"):
                    tf.summary.scalar("loss_generator_total", self.gen_loss_total, ["gen"])
                    tf.summary.scalar("loss_gen_adv", self.gen_adv_loss, ["gen"])
                    tf.summary.scalar("loss_gen_con", self.contextual_loss, ["gen"])
                    tf.summary.scalar("loss_gen_enc", self.latent_loss, ["gen"])
                with tf.name_scope("image_summary"):
                    tf.summary.image("reconstruct", self.img_rec, 5, ["image"])
                    tf.summary.image("input_images", self.image_input, 5, ["image"])

        if self.config.trainer.enable_early_stop:
            with tf.name_scope("validation_summary"):
                tf.summary.scalar("valid", self.rec_error_valid, ["v"])

        self.sum_op_dis = tf.summary.merge_all("dis")
        self.sum_op_gen = tf.summary.merge_all("gen")
        self.sum_op_im = tf.summary.merge_all("image")
        self.sum_op_valid = tf.summary.merge_all("v")

    def generator(self, image_input, getter=None):
        # Make the generator model
        with tf.variable_scope("Generator", custom_getter=getter, reuse=tf.AUTO_REUSE):
            # Encoder Part
            with tf.variable_scope("Encoder"):
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
            with tf.variable_scope("Decoder"):
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
                        activation=tf.nn.tanh,
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
