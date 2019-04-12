import tensorflow as tf

from base.base_model import BaseModel
import utils.alad_utils as sn


class ANOGAN(BaseModel):
    def __init__(self, config):
        super(ANOGAN, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # Placeholdersn
        self.is_training = tf.placeholder(tf.bool)
        self.image_tensor = tf.placeholder(
            tf.float32, shape=[None] + self.config.trainer.image_dims, name="x"
        )
        self.noise_tensor = tf.placeholder(
            tf.float32, shape=[None, self.config.trainer.noise_dim], name="noise"
        )
        # Placeholders for the true and fake labels
        self.true_labels = tf.placeholder(
            dtype=tf.float32, shape=[None, 1], name="true_labels"
        )
        self.generated_labels = tf.placeholder(
            dtype=tf.float32, shape=[None, 1], name="gen_labels"
        )
        # Building the Graph
        self.logger.info("Building Graph")
        with tf.variable_scope("ANOGAN"):
            # Generator
            self.img_gen = self.generator(self.noise_tensor)
            # Discriminator
            disc_real, inter_layer_real = self.discriminator(self.image_tensor)
            disc_fake, inter_layer_fake = self.discriminator(self.img_gen)

        # Losses of the training of Generator and Discriminator
        ########################################################################
        # METRICS
        ########################################################################
        with tf.variable_scope("Loss_Functions"):
            self.disc_loss_real = tf.reduce_mean(
                tf.losses.sigmoid_cross_entropy(
                    multi_class_labels=self.true_labels,
                    logits=disc_real,
                    scope="real_disc_loss",
                )
            )
            self.disc_loss_fake = tf.reduce_mean(
                tf.losses.sigmoid_cross_entropy(
                    multi_class_labels=self.generated_labels,
                    logits=disc_fake,
                    scope="fake_disc_loss",
                )
            )
            self.total_disc_loss = self.disc_loss_real + self.disc_loss_fake
            if self.config.trainer.soft_labels:
                labels = tf.zeros_like(disc_fake)
            else:
                labels = tf.ones_like(disc_fake)
            self.gen_loss = tf.reduce_mean(
                tf.losses.sigmoid_cross_entropy(
                    labels, disc_fake, scope="generator_loss"
                )
            )
        ########################################################################
        # OPTIMIZATION
        ########################################################################
        # Build the Optimizers
        with tf.variable_scope("Optimizers"):
            # Collect all the variables
            all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # Generator Network Variables
            self.generator_vars = [
                v for v in all_variables if v.name.startswith("ANOGAN/Generator")
            ]
            # Discriminator Network Variables
            self.discriminator_vars = [
                v for v in all_variables if v.name.startswith("ANOGAN/Discriminator")
            ]
            # Create Training Operations
            # Generator Network Operations
            self.gen_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="ANOGAN/Generator"
            )
            # Discriminator Network Operations
            self.disc_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="ANOGAN/Discriminator"
            )
            # Initialization of Optimizers
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
            with tf.control_dependencies(self.gen_update_ops):
                self.train_gen = self.generator_optimizer.minimize(
                    self.gen_loss,
                    global_step=self.global_step_tensor,
                    var_list=self.generator_vars,
                )
            with tf.control_dependencies(self.disc_update_ops):
                self.train_disc = self.discriminator_optimizer.minimize(
                    self.total_disc_loss,
                    global_step=self.global_step_tensor,
                    var_list=self.discriminator_vars,
                )

            def train_op_with_ema_dependency(vars, op):
                ema = tf.train.ExponentialMovingAverage(
                    decay=self.config.trainer.ema_decay
                )
                maintain_averages_op = ema.apply(vars)
                with tf.control_dependencies([op]):
                    train_op = tf.group(maintain_averages_op)
                return train_op, ema

            self.train_gen_op, self.gen_ema = train_op_with_ema_dependency(
                self.generator_vars, self.train_gen
            )
            self.train_dis_op, self.dis_ema = train_op_with_ema_dependency(
                self.discriminator_vars, self.train_disc
            )
        with tf.variable_scope("Latent_variable"):
            self.z_optim = tf.get_variable(
                name="z_optim",
                shape=[
                    self.config.data_loader.test_batch,
                    self.config.trainer.noise_dim,
                ],
                initializer=tf.truncated_normal_initializer(),
            )
            reinit_z = self.z_optim.initializer

        with tf.variable_scope("ANOGAN"):
            x_gen_ema = self.generator(
                self.noise_tensor, getter=sn.get_getter(self.gen_ema)
            )
            self.rec_gen_ema = self.generator(
                self.z_optim, getter=sn.get_getter(self.gen_ema)
            )
            # Pass real and fake images into discriminator separately
            real_d_ema, inter_layer_real_ema = self.discriminator(
                self.image_tensor, getter=sn.get_getter(self.dis_ema)
            )
            fake_d_ema, inter_layer_fake_ema = self.discriminator(
                self.rec_gen_ema, getter=sn.get_getter(self.dis_ema)
            )

        with tf.variable_scope("Testing"):
            with tf.variable_scope("Reconstruction_Loss"):
                delta = self.image_tensor - self.rec_gen_ema
                delta_flat = tf.layers.Flatten()(delta)
                self.reconstruction_score = tf.norm(
                    delta_flat,
                    ord=self.config.trainer.degree,
                    axis=1,
                    keepdims=False,
                    name="epsilon",
                )

            with tf.variable_scope("Discriminator_Scores"):
                # TODO only one score method
                dis_score = tf.losses.sigmoid_cross_entropy(
                    labels=tf.ones_like(fake_d_ema), logits=fake_d_ema
                )

                dis_score = tf.squeeze(dis_score)

            with tf.variable_scope("Score"):
                self.loss_invert = (
                    self.config.trainer.weight * self.reconstruction_score
                    + (1 - self.config.trainer.weight) * dis_score
                )

        self.rec_error_valid = tf.reduce_mean(self.loss_invert)

        with tf.variable_scope("Test_Learning_Rate"):
            step_lr = tf.Variable(0, trainable=False)
            learning_rate_invert = 0.001
            reinit_lr = tf.variables_initializer(
                tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope="Test_Learning_Rate"
                )
            )

        with tf.name_scope("Test_Optimizer"):
            self.invert_op = tf.train.AdamOptimizer(learning_rate_invert).minimize(
                self.loss_invert,
                global_step=step_lr,
                var_list=[self.z_optim],
                name="optimizer",
            )
            reinit_optim = tf.variables_initializer(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Test_Optimizer")
            )

        self.reinit_test_graph_op = [reinit_z, reinit_lr, reinit_optim]

        with tf.name_scope("Scores"):
            list_scores = self.loss_invert

        if self.config.log.enable_summary:
            with tf.name_scope("Training_Summary"):
                with tf.name_scope("Dis_Summary"):
                    tf.summary.scalar(
                        "Real_Discriminator_Loss", self.disc_loss_real, ["dis"]
                    )
                    tf.summary.scalar(
                        "Fake_Discriminator_Loss", self.disc_loss_fake, ["dis"]
                    )
                    tf.summary.scalar(
                        "Discriminator_Loss", self.total_disc_loss, ["dis"]
                    )

                with tf.name_scope("Gen_Summary"):
                    tf.summary.scalar("Loss_Generator", self.gen_loss, ["gen"])

            with tf.name_scope("Img_Summary"):
                heatmap_pl_latent = tf.placeholder(
                    tf.float32, shape=(1, 480, 640, 3), name="heatmap_pl_latent"
                )
                sum_op_latent = tf.summary.image("heatmap_latent", heatmap_pl_latent)

            with tf.name_scope("Validation_Summary"):
                tf.summary.scalar("valid", self.rec_error_valid, ["v"])

            with tf.name_scope("image_summary"):
                tf.summary.image("reconstruct", self.img_gen, 8, ["image"])
                tf.summary.image("input_images", self.image_tensor, 8, ["image"])

            self.sum_op_dis = tf.summary.merge_all("dis")
            self.sum_op_gen = tf.summary.merge_all("gen")
            self.sum_op = tf.summary.merge([self.sum_op_dis, self.sum_op_gen])
            self.sum_op_im = tf.summary.merge_all("image")
            self.sum_op_valid = tf.summary.merge_all("v")

    def generator(self, noise_tensor, getter=None):
        """ Generator architecture in tensorflow

        Generates the data from the latent space
        Args:
            noise_tensor: input variable in the latent space
            getter: for exponential moving average during inference
            reuse: sharing variables or not
        """
        with tf.variable_scope("Generator", reuse=tf.AUTO_REUSE, custom_getter=getter):
            net = tf.reshape(noise_tensor, [-1, self.config.trainer.noise_dim])
            net_name = "layer_1"
            with tf.variable_scope(net_name):
                net = tf.layers.dense(
                    net,
                    units=7 * 7 * 512,
                    kernel_initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=0.01
                    ),
                    name="fc",
                )
                net = tf.layers.batch_normalization(
                    inputs=net,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="dense/bn",
                )
                net = tf.nn.relu(features=net, name="dense/relu")
                net = tf.reshape(net, shape=[-1, 7, 7, 512])

            net_name = "layer_2"
            with tf.variable_scope(net_name):
                net = tf.layers.Conv2DTranspose(
                    filters=512,
                    kernel_size=5,
                    strides=(1, 1),
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=0.01
                    ),
                    name="tconv1",
                )(net)
                net = tf.layers.batch_normalization(
                    inputs=net,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="tconv1/bn",
                )
                net = tf.nn.relu(features=net, name="tconv1/relu")

            net_name = "layer_3"
            with tf.variable_scope(net_name):
                net = tf.layers.Conv2DTranspose(
                    filters=256,
                    kernel_size=5,
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=0.01
                    ),
                    name="tconv2",
                )(net)
                net = tf.layers.batch_normalization(
                    inputs=net,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="tconv2/bn",
                )
                net = tf.nn.relu(features=net, name="tconv2/relu")

            net_name = "layer_4"
            with tf.variable_scope(net_name):
                net = tf.layers.Conv2DTranspose(
                    filters=128,
                    kernel_size=5,
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=0.01
                    ),
                    name="tconv3",
                )(net)
                net = tf.layers.batch_normalization(
                    inputs=net,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="tconv3/bn",
                )
                net = tf.nn.relu(features=net, name="tconv3/relu")

            net_name = "layer_5"
            with tf.variable_scope(net_name):
                net = tf.layers.Conv2DTranspose(
                    filters=1,
                    kernel_size=5,
                    strides=(1, 1),
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=0.01
                    ),
                    name="tconv4",
                )(net)
                net = tf.nn.tanh(net, name="tconv4/tanh")

        return net

    def discriminator(self, img_tensor, getter=None):
        """ Discriminator architecture in tensorflow

        Discriminates between pairs (E(x), x) and (z, G(z))
        Args:
            img_tensor:
            noise_tensor:
            getter: for exponential moving average during inference
            reuse: sharing variables or not
            do_spectral_norm:
        """
        with tf.variable_scope(
            "Discriminator", reuse=tf.AUTO_REUSE, custom_getter=getter
        ):
            net_name = "layer_1"
            with tf.variable_scope(net_name):
                x = tf.layers.conv2d(
                    img_tensor,
                    filters=128,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=0.01
                    ),
                    name="conv1",
                )
                x = tf.nn.leaky_relu(
                    features=x,
                    alpha=self.config.trainer.leakyReLU_alpha,
                    name="conv1/leaky_relu",
                )

            net_name = "layer_2"
            with tf.variable_scope(net_name):
                x = tf.layers.conv2d(
                    x,
                    filters=256,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=0.01
                    ),
                    name="conv2",
                )
                x = tf.layers.batch_normalization(
                    inputs=x,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="tconv2/bn",
                )
                x = tf.nn.leaky_relu(
                    features=x,
                    alpha=self.config.trainer.leakyReLU_alpha,
                    name="conv2/leaky_relu",
                )
            net_name = "layer_3"
            with tf.variable_scope(net_name):
                x = tf.layers.conv2d(
                    x,
                    filters=512,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=0.01
                    ),
                    name="conv2",
                )
                x = tf.layers.batch_normalization(
                    inputs=x,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training,
                    name="tconv3/bn",
                )
                x = tf.nn.leaky_relu(
                    features=x,
                    alpha=self.config.trainer.leakyReLU_alpha,
                    name="conv3/leaky_relu",
                )

            net = tf.layers.Flatten()(x)

            intermediate_layer = x
            net_name = "layer_4"
            with tf.variable_scope(net_name):
                net = tf.layers.dense(
                    net,
                    units=1,
                    kernel_initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=0.01
                    ),
                    name="fc",
                )

            logits = tf.squeeze(net)

            return logits, intermediate_layer

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.log.max_to_keep)
