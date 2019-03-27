import tensorflow as tf

from base.base_model import BaseModel


class GAN(BaseModel):
    def __init__(self, config):
        super(GAN, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # Placeholders
        self.noise_tensor = tf.placeholder(
            tf.float32, shape=[None, self.config.noise_dim], name="noise"
        )
        self.image_input = tf.placeholder(
            tf.float32, shape=[None] + self.config.image_dims, name="x"
        )
        self.random_vector_for_generation = tf.random_normal(
            [self.config.num_example_imgs_to_generate, self.config.noise_dim],
            name="sampler",
        )
        # Random Noise addition to both image and the noise
        # This makes it harder for the discriminator to do it's job, preventing
        # it from always "winning" the GAN min/max contest
        self.real_noise = tf.placeholder(
            tf.float32, shape=[None] + self.config.image_dims, name="real_noise"
        )
        self.fake_noise = tf.placeholder(
            tf.float32, shape=[None] + self.config.image_dims, name="fake_noise"
        )

        self.real_image = self.image_input + self.real_noise

        self.initializer = tf.truncated_normal_initializer(stddev=0.02)
        # Make the Generator model
        ########################################################################
        # GENERATOR
        ########################################################################
        with tf.name_scope("Generator"):
            # Input layer creates the entry point to the model
            inputs_g = tf.keras.layers.Input(shape=[self.config.noise_dim])
            # Densely connected Neural Network layer with 12544 Neurons.
            x_g = tf.keras.layers.Dense(
                7 * 7 * 256, use_bias=False, kernel_initializer=self.initializer
            )(inputs_g)
            # Normalize the output of the Layer
            x_g = tf.keras.layers.BatchNormalization(
                momentum=self.config.batch_momentum
            )(x_g)
            # f(x) = alpha * x for x < 0, f(x) = x for x >= 0.
            x_g = tf.keras.layers.LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_g)
            # Reshaping the output
            x_g = tf.keras.layers.Reshape((7, 7, 256))(x_g)
            # Check the size of the current output just in case
            assert x_g.get_shape().as_list() == [None, 7, 7, 256]
            x_g = tf.keras.layers.Conv2DTranspose(
                128,
                (5, 5),
                strides=(1, 1),
                padding="same",
                use_bias=False   ,
                 kernel_initializer=self.initializer,
            )(x_g)
            assert x_g.get_shape().as_list() == [None, 7, 7, 128]
            x_g = tf.keras.layers.BatchNormalization(
                momentum=self.config.batch_momentum
            )(x_g)
            x_g = tf.keras.layers.LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_g)

            x_g = tf.keras.layers.Conv2DTranspose(
                64,
                (5, 5),
                strides=(2, 2),
                padding="same",
                use_bias=False   ,
                 kernel_initializer=self.initializer,
            )(x_g)
            assert x_g.get_shape().as_list() == [None, 14, 14, 64]
            x_g = tf.keras.layers.BatchNormalization(
                momentum=self.config.batch_momentum
            )(x_g)
            x_g = tf.keras.layers.LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_g)

            x_g = tf.keras.layers.Conv2DTranspose(
                1,
                (5, 5),
                strides=(2, 2),
                padding="same",
                use_bias=False,
                activation="tanh"   ,
                 kernel_initializer=self.initializer,
            )(x_g)
            assert x_g.get_shape().as_list() == [None, 28, 28, 1]
            self.generator = tf.keras.models.Model(inputs=inputs_g, outputs=x_g)

        # Make the discriminator model
        ########################################################################
        # DISCRIMINATOR
        ########################################################################
        with tf.name_scope("Discriminator"):
            inputs_d = tf.keras.layers.Input(shape=self.config.image_dims)
            x_d = tf.keras.layers.Conv2D(
                32,
                (5, 5),
                strides=(2, 2),
                padding="same"   ,
                 kernel_initializer=self.initializer,
            )(inputs_d)
            x_d = tf.keras.layers.LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_d)
            x_d = tf.keras.layers.Dropout(rate=self.config.dropout_rate)(x_d)
            x_d = tf.keras.layers.Conv2D(
                64,
                (5, 5),
                strides=(2, 2),
                padding="same"   ,
                 kernel_initializer=self.initializer,
            )(x_d)
            x_d = tf.keras.layers.LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_d)
            x_d = tf.keras.layers.Dropout(rate=self.config.dropout_rate)(x_d)
            x_d = tf.keras.layers.Flatten()(x_d)
            x_d = tf.keras.layers.Dense(1)(x_d)
            self.discriminator = tf.keras.models.Model(inputs=inputs_d, outputs=x_d)
        # Evaluations for the training
        # Adding noise part is new
        generated_image = (
            self.generator(self.noise_tensor, training=True) + self.fake_noise
        )
        real_output = self.discriminator(self.real_image, training=True)
        generated_output = self.discriminator(generated_image, training=True)

        # Losses of the training of Generator and Discriminator
        ########################################################################
        # METRICS
        ########################################################################
        with tf.name_scope("Generator_Loss"):
            self.gen_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
                tf.ones_like(generated_output), generated_output
            ))
        with tf.name_scope("Discriminator_Loss"):
            self.disc_real_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
                multi_class_labels=tf.ones_like(real_output), logits=real_output
            ))
            self.disc_gen_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
                multi_class_labels=tf.zeros_like(generated_output),
                logits=generated_output,
            ))
            self.total_disc_loss = self.disc_real_loss + self.disc_gen_loss
        # Accuracy of the model
        with tf.name_scope("Accuracy"):
            self.accuracy_fake = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        tf.round(generated_output), tf.zeros_like(generated_output)
                    ),
                    tf.float32,
                )
            )
            self.accuracy_real = tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.round(real_output), tf.ones_like(real_output)),
                    tf.float32,
                )
            )
            self.accuracy_total = 0.5 * (self.accuracy_fake + self.accuracy_real)

        # Store the loss values for the Tensorboard
        ########################################################################
        # TENSORBOARD
        ########################################################################
        tf.summary.scalar("Generator_Loss", self.gen_loss)
        tf.summary.scalar("Discriminator_Real_Loss", self.disc_real_loss)
        tf.summary.scalar("Real_Accuracy", self.accuracy_real)
        tf.summary.scalar("Fake_Accuracy", self.accuracy_fake)
        tf.summary.scalar("Total_Accuracy", self.accuracy_total)
        tf.summary.scalar("Discriminator_Gen_Loss", self.disc_gen_loss)
        tf.summary.scalar("Discriminator_Total_Loss", self.total_disc_loss)
        # Images for the Tensorboard
        tf.summary.image("From_Noise", tf.reshape(generated_image, [-1, 28, 28, 1]))
        tf.summary.image("Real_Image", tf.reshape(self.image_input, [-1, 28, 28, 1]))
        # Sample Operation
        with tf.name_scope("Generator_Progress"):
            self.progress_images = self.generator(self.noise_tensor, training=False)

        ########################################################################
        # OPTIMIZATION
        ########################################################################
        all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.generator_vars = [v for v in all_variables if v.name.startswith("Generator")]
        self.discriminator_vars = [v for v in all_variables if v.name.startswith("Discriminator")]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # Initialization of Optimizers
        with tf.control_dependencies(update_ops):
            self.generator_optimizer = tf.train.AdamOptimizer(
                self.config.generator_l_rate
            )
            self.discriminator_optimizer = tf.train.AdamOptimizer(
                self.config.discriminator_l_rate
            )

        with tf.name_scope("SGD_Generator"):
            self.train_gen = self.generator_optimizer.minimize(
                self.gen_loss, global_step=self.global_step_tensor,
                var_list=self.generator_vars
            )

        with tf.name_scope("SGD_Discriminator"):
            self.train_disc = self.discriminator_optimizer.minimize(
                self.total_disc_loss, global_step=self.global_step_tensor,
                var_list=self.discriminator_vars
            )



        for i in range(0, 10):
            with tf.name_scope("layer" + str(i)):
                pesos = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                tf.summary.histogram("pesos" + str(i), pesos[i])
        self.summary = tf.summary.merge_all()

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
