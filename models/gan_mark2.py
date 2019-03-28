
import tensorflow as tf

from base.base_model import BaseModel
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D,UpSampling2D


class GAN_mark2(BaseModel):
    def __init__(self, config):
        super(GAN_mark2, self).__init__(config)
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
        # Make the Generator Model
        with tf.name_scope("Generator"):
            # Keras Model Object is Defined
            model_g = Sequential()
            model_g.add(Dense(128 * 7 * 7, activation="relu", input_shape=[self.config.noise_dim]))
            model_g.add(Reshape((7, 7, 128)))
            model_g.add(UpSampling2D())
            model_g.add(Conv2D(128, kernel_size=3,strides=(1, 1), padding='same'))
            model_g.add(BatchNormalization(momentum=self.config.batch_momentum))
            model_g.add(Activation("relu"))
            model_g.add(UpSampling2D())
            model_g.add(Conv2D(64, kernel_size=3,strides=(1, 1), padding='same'))
            model_g.add(BatchNormalization(momentum=self.config.batch_momentum))
            model_g.add(Activation("relu"))
            model_g.add(Conv2D(1, kernel_size=3,padding='same'))
            model_g.add(Activation("tanh"))
            assert model_g.output_shape == (None, 28, 28, 1)
            model_g.summary()
            input_g = Input(shape=[self.config.noise_dim])
            output_img_g = model_g(input_g)

            self.generator = Model(input_g, output_img_g)

        # Make the Discriminator Model
        with tf.name_scope("Discriminator"):
            # Keras Model object is defined
            model_d = Sequential()
            model_d.add(Conv2D(32, kernel_size=3, strides=(2, 2),
                             input_shape=self.config.image_dims, padding="same"))
            model_d.add(LeakyReLU(alpha=self.config.leakyReLU_alpha))
            model_d.add(Dropout(rate=self.config.dropout_rate))
            model_d.add(Conv2D(64, kernel_size=3, strides=(2, 2), padding="same"))
            model_d.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
            model_d.add(BatchNormalization(momentum=self.config.batch_momentum))
            model_d.add(LeakyReLU(alpha=self.config.leakyReLU_alpha))
            model_d.add(Dropout(rate=self.config.dropout_rate))
            model_d.add(Conv2D(128, kernel_size=3, strides=(2, 2), padding="same"))
            model_d.add(BatchNormalization(momentum=self.config.batch_momentum))
            model_d.add(LeakyReLU(alpha=self.config.leakyReLU_alpha))
            model_d.add(Dropout(rate=self.config.dropout_rate))
            model_d.add(Conv2D(256, kernel_size=3, strides=(1, 1), padding="same"))
            model_d.add(BatchNormalization(momentum=self.config.batch_momentum))
            model_d.add(LeakyReLU(alpha=self.config.leakyReLU_alpha))
            model_d.add(Dropout(rate=self.config.dropout_rate))
            model_d.add(Flatten())
            model_d.add(Dense(1, activation='sigmoid'))

            model_d.summary()
            input_d = Input(shape=self.config.image_dims)
            output_validity = model_d(input_d)

            self.discriminator = Model(input_d, output_validity)

        # Train part of the Model
        # Get the Noise and generate a batch of New Images
        with tf.name_scope("Generated_Image"):
            generated_image = self.generator(self.noise_tensor,training=True) + self.fake_noise

        # Feed the Real image to the Discriminator and obtain the result
        with tf.name_scope("Real_From_Disc"):
            real_output = self.discriminator(self.real_image, training=True)
        # Feed the Generated image to the Discriminator and obtain the result
        with tf.name_scope("Gen_From_Disc"):
            generated_output = self.discriminator(generated_image,training=True)
        # Define the loss function of Discriminator
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

        # Variable Saving for the Tensorboard
        tf.summary.scalar("Generator_Loss", self.gen_loss)
        tf.summary.scalar("Discriminator_Loss", self.total_disc_loss)
        tf.summary.image('From_Noise', tf.reshape(
            generated_image, [-1, 28, 28, 1]))
        tf.summary.image('Real_Image', tf.reshape(
            self.real_image, [-1, 28, 28, 1]))
        # Define the operation to generate the test images from the current Generator Network
        with tf.name_scope("Generator_Progress"):
            self.progress_images = self.generator(self.noise_tensor,training=False)
        ########################################################################
        # OPTIMIZATION
        ########################################################################
        all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.generator_vars = [v for v in all_variables if v.name.startswith("Generator")]
        self.discriminator_vars = [v for v in all_variables if v.name.startswith("Discriminator")]
        # Initialize and Connect the Optimizers
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.gen_optimizer = tf.train.AdamOptimizer(
                learning_rate =self.config.generator_l_rate,beta1=self.config.optimizer_adam_beta1)
            self.disc_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.discriminator_l_rate,beta1=self.config.optimizer_adam_beta1)

        # Make the minimization of the Models
        # If there needs to be proprocessing for the gradients before applying them, use
        # Compute_gradients then apply_gradients
        with tf.name_scope('SGD_Gen'):
            self.train_gen = self.gen_optimizer.minimize(
                self.gen_loss,global_step=self.global_step_tensor,var_list=self.generator_vars)
        with tf.name_scope('SGD_Disc'):
            self.train_disc = self.disc_optimizer.minimize(
                self.total_disc_loss,global_step=self.global_step_tensor,var_list=self.discriminator_vars)

        
        # Histogram for the Discriminator Network
        for i in range(0, 18):
            with tf.name_scope('Disc_layer' + str(i)):
                pesos = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                tf.summary.histogram('pesos' + str(i), pesos[i])
        
        self.summary = tf.summary.merge_all()



    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

