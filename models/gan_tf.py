import tensorflow as tf

from base.base_model import BaseModel


class GAN_TF(BaseModel):
    def __init__(self, config):
        super(GAN_TF, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # Placeholders
        self.is_training = tf.placeholder(tf.bool)
        self.image_input = tf.placeholder(
            tf.float32, shape=[None] + self.config.trainer.image_dims, name="x"
        )
        self.noise_tensor = tf.placeholder(
            tf.float32, shape=[None, self.config.trainer.noise_dim], name="noise"
        )
        self.init_kernel = tf.truncated_normal_initializer(stddev=0.02)
        # Random Noise addition to both image and the noise
        # This makes it harder for the discriminator to do it's job, preventing
        # it from always "winning" the GAN min/max contest
        self.real_noise = tf.placeholder(
            dtype=tf.float32,
            shape=[None] + self.config.trainer.image_dims,
            name="real_noise",
        )
        self.fake_noise = tf.placeholder(
            dtype=tf.float32,
            shape=[None] + self.config.trainer.image_dims,
            name="fake_noise",
        )

        # self.real_image = self.image_input + self.fake_noise
        # Placeholders for the true and fake labels
        self.true_labels = tf.placeholder(
            dtype=tf.float32, shape=[None, 1], name="true_labels"
        )
        self.generated_labels = tf.placeholder(
            dtype=tf.float32, shape=[None, 1], name="gen_labels"
        )
        # Full Model Scope
        self.logger.info("Building Graph...")
        with tf.variable_scope("DCGAN"):
            self.generated_sample = self.generator(self.noise_tensor) + self.fake_noise
            disc_real = self.discriminator(self.image_input + self.real_noise)
            disc_fake = self.discriminator(self.generated_sample, reuse=True)
            # self.sample_image = self.generator(self.noise_tensor, reuse=True)
            self.stacked_gan = self.discriminator(self.generated_sample, reuse=True)
            # Losses of the training of Generator and Discriminator
            ########################################################################
            # METRICS
            ########################################################################

        with tf.name_scope("Discriminator_Loss"):
            self.disc_loss_real = tf.reduce_mean(
                tf.losses.sigmoid_cross_entropy(
                    multi_class_labels=self.true_labels, logits=disc_real
                )
            )
            self.disc_loss_fake = tf.reduce_mean(
                tf.losses.sigmoid_cross_entropy(
                    multi_class_labels=self.generated_labels, logits=disc_fake
                )
            )
            # Sum the both losses
            self.total_disc_loss = self.disc_loss_real + self.disc_loss_fake

        with tf.name_scope("Generator_Loss"):

            if self.config.soft_labels:
                self.gen_loss = tf.reduce_mean(
                    tf.losses.sigmoid_cross_entropy(
                        tf.zeros_like(self.stacked_gan), self.stacked_gan
                    )
                )
            else:
                self.gen_loss = tf.reduce_mean(
                    tf.losses.sigmoid_cross_entropy(
                        tf.ones_like(self.stacked_gan), self.stacked_gan
                    )
                )

        # Store the loss values for the Tensorboard
        ########################################################################
        # TENSORBOARD
        ########################################################################
        s_gen_loss = tf.summary.scalar("Generator_Loss", self.gen_loss)
        s_gen_img = tf.summary.image(
            "From_Noise", tf.reshape(self.generated_sample, [-1, 28, 28, 1])
        )
        self.summary_gan = tf.summary.merge(inputs=[s_gen_loss, s_gen_img])
        s_disc_r = tf.summary.scalar("Discriminator_Real_Loss", self.disc_loss_real)
        s_disc_f = tf.summary.scalar("Discriminator_Gen_Loss", self.disc_loss_fake)
        s_disc_t = tf.summary.scalar("Discriminator_Total_Loss", self.total_disc_loss)
        s_disc_img = tf.summary.image(
            "Real_Image", tf.reshape(self.image_input, [-1, 28, 28, 1])
        )
        self.summary_disc = tf.summary.merge(
            inputs=[s_disc_r, s_disc_f, s_disc_t, s_disc_img]
        )

        # Sample Operation

        ########################################################################
        # OPTIMIZATION
        ########################################################################
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
        # Collect all the variables
        all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # Generator Network Variables
        self.generator_vars = [
            v for v in all_variables if v.name.startswith("DCGAN/Generator")
        ]
        # Discriminator Network Variables
        self.discriminator_vars = [
            v for v in all_variables if v.name.startswith("DCGAN/Discriminator")
        ]
        # Create Training Operations
        # Generator Network Operations
        self.gen_update_ops = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS, scope="DCGAN/Generator"
        )
        # Discriminator Network Operations
        self.disc_update_ops = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS, scope="DCGAN/Discriminator"
        )
        # Initialization of Optimizers

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

        for i in range(0, 10):
            with tf.name_scope("layer" + str(i)):
                pesos = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                tf.summary.histogram("pesos" + str(i), pesos[i])

        # self.summary = tf.summary.merge_all()

    def generator(self, noise_tensor, reuse=False):
        # Make the Generator model
        with tf.variable_scope("Generator", reuse=reuse) as scope:
            # Densely connected Neural Network layer with 12544 Neurons.
            x_g = tf.layers.Dense(
                units=7 * 7 * 256,
                use_bias=False,
                kernel_initializer=self.init_kernel,
                name="g_dense",
            )(noise_tensor)
            # Normalize the output of the Layer

            x_g = tf.layers.batch_normalization(
                inputs=x_g,
                momentum=self.config.trainer.batch_momentum,
                training=self.is_training,
                name="g_bn_1",
            )

            # f(x) = alpha * x for x < 0, f(x) = x for x >= 0.
            x_g = tf.nn.leaky_relu(
                features=x_g, alpha=self.config.trainer.leakyReLU_alpha, name="g_lr_1"
            )
            # Reshaping the output
            x_g = tf.reshape(x_g, shape=[-1, 7, 7, 256])
            # Check the size of the current output just in case
            assert x_g.get_shape().as_list() == [None, 7, 7, 256]
            # First Conv2DTranspose Layer
            x_g = tf.layers.Conv2DTranspose(
                filters=128,
                kernel_size=5,
                strides=(1, 1),
                padding="same",
                use_bias=False,
                kernel_initializer=self.init_kernel,
                name="g_conv2dtr_1",
            )(x_g)
            assert x_g.get_shape().as_list() == [None, 7, 7, 128]

            x_g = tf.layers.batch_normalization(
                inputs=x_g,
                momentum=self.config.trainer.batch_momentum,
                training=self.is_training,
                name="g_bn_2",
            )
            x_g = tf.nn.leaky_relu(
                features=x_g, alpha=self.config.trainer.leakyReLU_alpha, name="g_lr_2"
            )
            # Second Conv2DTranspose Layer
            x_g = tf.layers.Conv2DTranspose(
                filters=128,
                kernel_size=5,
                strides=(2, 2),
                padding="same",
                use_bias=False,
                kernel_initializer=self.init_kernel,
                name="g_conv2dtr_2",
            )(x_g)
            assert x_g.get_shape().as_list() == [None, 14, 14, 128]

            x_g = tf.layers.batch_normalization(
                inputs=x_g,
                momentum=self.config.trainer.batch_momentum,
                training=self.is_training,
                name="g_bn_3",
            )
            x_g = tf.nn.leaky_relu(
                features=x_g, alpha=self.config.trainer.leakyReLU_alpha, name="g_lr_3"
            )
            # Third Conv2DTranspose Layer
            x_g = tf.layers.Conv2DTranspose(
                filters=128,
                kernel_size=5,
                strides=(2, 2),
                padding="same",
                use_bias=False,
                kernel_initializer=self.init_kernel,
                name="g_conv2dtr_3",
            )(x_g)
            assert x_g.get_shape().as_list() == [None, 28, 28, 128]

            x_g = tf.layers.batch_normalization(
                inputs=x_g,
                momentum=self.config.trainer.batch_momentum,
                training=self.is_training,
                name="g_bn_4",
            )
            x_g = tf.nn.leaky_relu(
                features=x_g, alpha=self.config.trainer.leakyReLU_alpha, name="g_lr_4"
            )
            # Final Conv2DTranspose Layer
            x_g = tf.layers.Conv2DTranspose(
                filters=1,
                kernel_size=5,
                strides=(1, 1),
                padding="same",
                use_bias=False,
                activation=tf.nn.tanh,
                kernel_initializer=self.init_kernel,
                name="g_conv2dtr_4",
            )(x_g)
            assert x_g.get_shape().as_list() == [None, 28, 28, 1]
            return x_g

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("Discriminator", reuse=reuse) as scope:
            # First Convolutional Layer
            x_d = tf.layers.Conv2D(
                filters=128,
                kernel_size=5,
                strides=(1, 1),
                padding="same",
                kernel_initializer=self.init_kernel,
                name="d_conv1",
            )(image)
            x_d = tf.layers.batch_normalization(
                inputs=x_d,
                momentum=self.config.trainer.batch_momentum,
                training=self.is_training,
                name="d_bn_1",
            )
            x_d = tf.nn.leaky_relu(
                features=x_d, alpha=self.config.trainer.leakyReLU_alpha, name="d_lr_1"
            )
            # Second Convolutional Layer
            x_d = tf.layers.Conv2D(
                filters=64,
                kernel_size=5,
                strides=(2, 2),
                padding="same",
                kernel_initializer=self.init_kernel,
                name="d_conv_2",
            )(x_d)
            x_d = tf.layers.batch_normalization(
                inputs=x_d,
                momentum=self.config.trainer.batch_momentum,
                training=self.is_training,
                name="d_bn_2",
            )
            x_d = tf.nn.leaky_relu(
                features=x_d, alpha=self.config.trainer.leakyReLU_alpha, name="d_lr_2"
            )

            # Third Convolutional Layer
            x_d = tf.layers.Conv2D(
                filters=32,
                kernel_size=5,
                strides=(2, 2),
                padding="same",
                kernel_initializer=self.init_kernel,
                name="d_conv_3",
            )(x_d)
            x_d = tf.layers.batch_normalization(
                inputs=x_d,
                momentum=self.config.trainer.batch_momentum,
                training=self.is_training,
                name="d_bn_3",
            )
            x_d = tf.nn.leaky_relu(
                features=x_d, alpha=self.config.trainer.leakyReLU_alpha, name="d_lr_3"
            )

            x_d = tf.layers.Flatten(name="d_flatten")(x_d)
            x_d = tf.layers.Dropout(
                rate=self.config.trainer.dropout_rate, name="d_dropout"
            )(x_d)
            x_d = tf.layers.Dense(units=1, name="d_dense")(x_d)
            return x_d

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.log.max_to_keep)
