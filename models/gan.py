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
            dtype=tf.float32, shape=[None] + self.config.image_dims, name="real_noise"
        )
        self.fake_noise = tf.placeholder(
            dtype=tf.float32, shape=[None] + self.config.image_dims, name="fake_noise"
        )

        self.real_image = self.image_input + self.real_noise
        # Placeholders for the true and fake labels
        self.true_labels = tf.placeholder(dtype=tf.float32, shape=[None, 1],name="true_labels")
        self.generated_labels = tf.placeholder(dtype=tf.float32, shape=[None, 1],name="gen_labels")
        
        # Make the Generator model
        ########################################################################
        # GENERATOR
        ########################################################################
        with tf.variable_scope("Generator"):
            # Input layer creates the entry point to the model
            inputs_g = tf.keras.layers.Input(shape=[self.config.noise_dim])
            # Densely connected Neural Network layer with 12544 Neurons.
            x_g = tf.keras.layers.Dense(7 * 7 * 256, activation="relu",use_bias=False, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))(inputs_g)
            # Normalize the output of the Layer
            x_g = tf.keras.layers.BatchNormalization(momentum=self.config.batch_momentum)(x_g)
            # f(x) = alpha * x for x < 0, f(x) = x for x >= 0.
            x_g = tf.keras.layers.LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_g)
            # Reshaping the output
            x_g = tf.keras.layers.Reshape((7, 7, 256))(x_g)
            # Check the size of the current output just in case
            assert x_g.get_shape().as_list() == [None, 7, 7, 256]

            x_g = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))(x_g)
            assert x_g.get_shape().as_list() == [None, 7, 7, 128]
            x_g = tf.keras.layers.BatchNormalization(momentum=self.config.batch_momentum)(x_g)
            x_g = tf.keras.layers.LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_g)

            x_g = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", use_bias=False, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))(x_g)
            assert x_g.get_shape().as_list() == [None, 14, 14, 128]
            x_g = tf.keras.layers.BatchNormalization(momentum=self.config.batch_momentum)(x_g)
            x_g = tf.keras.layers.LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_g)

            x_g = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))(x_g)
            assert x_g.get_shape().as_list() == [None, 14, 14, 128]
            x_g = tf.keras.layers.BatchNormalization(momentum=self.config.batch_momentum)(x_g)
            x_g = tf.keras.layers.LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_g)

            x_g = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", use_bias=False, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))(x_g)
            assert x_g.get_shape().as_list() == [None, 28, 28, 128]
            x_g = tf.keras.layers.BatchNormalization(momentum=self.config.batch_momentum)(x_g)
            x_g = tf.keras.layers.LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_g)

            x_g = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding="same", use_bias=False, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),)(x_g)
            out = tf.keras.layers.Activation("tanh")(x_g)
            assert x_g.get_shape().as_list() == [None, 28, 28, 1]
            self.generator = tf.keras.models.Model(inputs=inputs_g, outputs=out)

        # Make the discriminator model
        ########################################################################
        # DISCRIMINATOR
        ########################################################################
        with tf.variable_scope("Discriminator"):
            inputs_d = tf.keras.layers.Input(shape=self.config.image_dims)
            # First Convolutional Layer
            x_d = tf.keras.layers.Conv2D(128, (5, 5), strides=(1, 1), padding="same",
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))(inputs_d)
            x_d = tf.keras.layers.BatchNormalization(momentum=self.config.batch_momentum)(x_d)
            x_d = tf.keras.layers.LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_d)
            #x_d = tf.keras.layers.Dropout(rate=self.config.dropout_rate)(x_d)
            # Second Convolutional Layer
            x_d = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same",
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))(x_d)
            x_d = tf.keras.layers.BatchNormalization(momentum=self.config.batch_momentum)(x_d)
            x_d = tf.keras.layers.LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_d)
            #x_d = tf.keras.layers.Dropout(rate=self.config.dropout_rate)(x_d)
            # Third Convolutional Layer
            x_d = tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1), padding="same",
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))(x_d)
            x_d = tf.keras.layers.BatchNormalization(momentum=self.config.batch_momentum)(x_d)
            x_d = tf.keras.layers.LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_d)
            #x_d = tf.keras.layers.Dropout(rate=self.config.dropout_rate)(x_d)
            # Fourth Convolutional Layer
            x_d = tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding="same",
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))(x_d)
            x_d = tf.keras.layers.BatchNormalization(momentum=self.config.batch_momentum)(x_d)
            x_d = tf.keras.layers.LeakyReLU(alpha=self.config.leakyReLU_alpha)(x_d)
            #x_d = tf.keras.layers.Dropout(rate=self.config.dropout_rate)(x_d)
            x_d = tf.keras.layers.Flatten()(x_d)
            x_d = tf.keras.layers.Dropout(rate=self.config.dropout_rate)(x_d)
            x_d = tf.keras.layers.Dense(1)(x_d)
            self.discriminator = tf.keras.models.Model(inputs=inputs_d, outputs=x_d)
        # Evaluations for the training
        # Adding noise part is new
        # Build the generator Network
        generated_sample = (
            self.generator(self.noise_tensor, training=True) + self.fake_noise
        )
        # Build 2 Discriminator Networks
        disc_real = self.discriminator(self.real_image, training=True)
        disc_fake = self.discriminator(generated_sample, training=True)
        # Build the stacked Generator/Discriminator
        stacked_gan = self.discriminator(generated_sample,training=True)
        # Losses of the training of Generator and Discriminator
        ########################################################################
        # METRICS
        ########################################################################

        with tf.name_scope("Discriminator_Loss"):
            self.disc_loss_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
                multi_class_labels=self.true_labels, logits=disc_real
            ))
            self.disc_loss_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
                multi_class_labels=self.generated_labels,
                logits=disc_fake,
            ))
            # Sum the both losses
            self.total_disc_loss = self.disc_loss_real + self.disc_loss_fake

        with tf.name_scope("Generator_Loss"):
            self.gen_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
                tf.zeros_like(stacked_gan), stacked_gan))

        # Accuracy of the model
        with tf.name_scope("Accuracy"):
            self.accuracy_fake = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        tf.round(disc_fake), tf.zeros_like(disc_fake)
                    ),
                    tf.float32,
                )
            )
            self.accuracy_real = tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.round(disc_real), tf.ones_like(disc_real)),
                    tf.float32,
                )
            )
            self.accuracy_total = 0.5 * (self.accuracy_fake + self.accuracy_real)

        # Store the loss values for the Tensorboard
        ########################################################################
        # TENSORBOARD
        ########################################################################
        tf.summary.scalar("Generator_Loss", self.gen_loss)
        tf.summary.scalar("Discriminator_Real_Loss", self.disc_loss_real)
        tf.summary.scalar("Real_Accuracy", self.accuracy_real)
        tf.summary.scalar("Fake_Accuracy", self.accuracy_fake)
        tf.summary.scalar("Total_Accuracy", self.accuracy_total)
        tf.summary.scalar("Discriminator_Gen_Loss", self.disc_loss_fake)
        tf.summary.scalar("Discriminator_Total_Loss", self.total_disc_loss)
        # Images for the Tensorboard
        tf.summary.image("From_Noise", tf.reshape(generated_sample, [-1, 28, 28, 1]))
        tf.summary.image("Real_Image", tf.reshape(self.image_input, [-1, 28, 28, 1]))
        # Sample Operation
        with tf.name_scope("Generator_Progress"):
            self.progress_images = self.generator(self.noise_tensor, training=False)

        ########################################################################
        # OPTIMIZATION
        ########################################################################
        # Build the Optimizers
        self.generator_optimizer = tf.train.AdamOptimizer(
            self.config.generator_l_rate,
            beta1=self.config.optimizer_adam_beta1,
            beta2=self.config.optimizer_adam_beta2
        )
        self.discriminator_optimizer = tf.train.AdamOptimizer(
            self.config.discriminator_l_rate,
            beta1=self.config.optimizer_adam_beta1,
            beta2=self.config.optimizer_adam_beta2
        )
        # Collect all the variables
        all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # Generator Network Variables
        self.generator_vars = [v for v in all_variables if v.name.startswith("Generator")]
        # Discriminator Network Variables
        self.discriminator_vars = [v for v in all_variables if v.name.startswith("Discriminator")]
        # Create Training Operations
        # Generator Network Operations
        self.gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Generator")
        # Discriminator Network Operations
        self.disc_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Discriminator")
        # Initialization of Optimizers
        with tf.control_dependencies(self.gen_update_ops):
            self.train_gen = self.generator_optimizer.minimize(
                self.gen_loss, global_step=self.global_step_tensor,
                var_list=self.generator_vars
            )
        with tf.control_dependencies(self.disc_update_ops):
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
