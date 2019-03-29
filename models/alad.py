
from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Reshape
from tensorflow.keras.layers import Conv2DTranspose, ReLU, Dropout
from tensorflow.keras.layers import Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

class ALAD(BaseModel):
    def __init__(self, config):
        super(ALAD, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        # Placeholders for the data
        # Noise placeholder [None, 100]
        self.noise_tensor = tf.placeholder(
            tf.float32,shape=[None, self.config.noise_dim]
        )
        # Real Image place holder [None, 28, 28, 1]
        self.image_output = tf.placeholder(
            tf.float32, shape=[None] + self.config.image_dims
        )
        # Kernel Initialization
        init_kernel = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        # Encoder Model TODO
        # Maps the data into the Latent Space
        with tf.variable_scope("Encoder"):

            with tf.variable_scope("Layer_1"):

                inputs_e = Input(shape=self.config.image_dims)
                layer_e =  Conv2D(filters=128,kernel_size=4, strides=(2, 2), padding="same",
                              kernel_initializer=init_kernel)(inputs_e)
                layer_e = BatchNormalization(momentum=self.config.batch_momentum)(layer_e)
                layer_e = LeakyReLU(alpha=self.config.leakyReLU_alpha)(layer_e)

            with tf.variable_scope("Layer_2"):

                layer_e = Conv2D(filters=256, kernel_size=4, strides=(2, 2), padding="same",
                                 kernel_initializer=init_kernel)(layer_e)
                layer_e = BatchNormalization(momentum=self.config.batch_momentum)(layer_e)
                layer_e = LeakyReLU(alpha=self.config.leakyReLU_alpha)(layer_e)

            with tf.variable_scope("Layer_3"):

                layer_e = Conv2D(filters=512, kernel_size=4, strides=(2, 2), padding="same",
                                 kernel_initializer=init_kernel)(layer_e)
                layer_e = BatchNormalization(momentum=self.config.batch_momentum)(layer_e)
                layer_e = LeakyReLU(alpha=self.config.leakyReLU_alpha)(layer_e)

            with tf.variable_scope("Layer_4"):

                layer_e = Conv2D(filters=self.config.noise_dim, kernel_size=4, strides=(1, 1), padding="valid",
                                 kernel_initializer=init_kernel)(layer_e)
                layer_e = Reshape((self.config.noise_dim,))(layer_e)
                self.encoder = Model(inputs=inputs_e,outputs=layer_e)


        # Generator Model TODO
        # Generate data from the latent space
        with tf.variable_scope("Generator"):

            inputs_g = Input(shape=[self.config.noise_dim])
            layer_g = Reshape((1, 1, self.config.noise_dim))(inputs_g)

            with tf.variable_scope("Layer_1"):

                layer_g = Conv2DTranspose(filters=512, kernel_size=4, strides=(2, 2), padding="valid",
                                          kernel_initializer=init_kernel)(layer_g)
                layer_g = BatchNormalization(momentum=self.config.batch_momentum)(layer_g)
                layer_g = ReLU()(layer_g)

            with tf.variable_scope("Layer_2"):
                layer_g = Conv2DTranspose(filters=256, kernel_size=4, strides=(2, 2), padding="same",
                                          kernel_initializer=init_kernel)(layer_g)
                layer_g = BatchNormalization(momentum=self.config.batch_momentum)(layer_g)
                layer_g = ReLU()(layer_g)

            with tf.variable_scope("Layer_3"):
                layer_g = Conv2DTranspose(filters=128, kernel_size=4, strides=(2, 2), padding="same",
                                          kernel_initializer=init_kernel)(layer_g)
                layer_g = BatchNormalization(momentum=self.config.batch_momentum)(layer_g)
                layer_g = ReLU()(layer_g)

            with tf.variable_scope("Layer_4"):
                layer_g = Conv2DTranspose(filters=1, kernel_size=4, strides=(2, 2), padding="same",
                                          activation="tanh",
                                          kernel_initializer=init_kernel)(layer_g)
            self.generator = Model(inputs=inputs_g, outputs=layer_g)


        # Discriminator model for xz TODO
        # Discriminates between pairs (E(x), x) and (G(z), z)
        with tf.variable_scope("Discriminator_xz"):

            inputs_x = Input(shape=self.config.image_dims, name="image_input")
            inputs_z = Input(shape=[self.config.noise_dim], name="noise_input")

            with tf.variable_scope("X_Layer_1"):

                layer_dx = Conv2D(filters=128,kernel_size=4, strides=(2, 2), padding="same",
                              kernel_initializer=init_kernel)(inputs_x)
                layer_dx = LeakyReLU(alpha=self.config.leakyReLU_alpha)(layer_dx)

            with tf.variable_scope("X_Layer_2"):

                layer_dx = Conv2D(filters=256, kernel_size=4, strides=(2, 2), padding="same",
                                  kernel_initializer=init_kernel)(layer_dx)
                layer_dx = BatchNormalization(momentum=self.config.batch_momentum)(layer_dx)
                layer_dx = LeakyReLU(alpha=self.config.leakyReLU_alpha)(layer_dx)

            with tf.variable_scope("X_Layer_3"):

                layer_dx = Conv2D(filters=512, kernel_size=4, strides=(2, 2), padding="same",
                                  kernel_initializer=init_kernel)(layer_dx)
                layer_dx = BatchNormalization(momentum=self.config.batch_momentum)(layer_dx)
                layer_dx = LeakyReLU(alpha=self.config.leakyReLU_alpha)(layer_dx)

                layer_dx = Reshape((1, 1, 512 * 4 * 4))(layer_dx)

            layer_dz = Reshape((1, 1, self.config.noise_dim))(inputs_z)

            with tf.variable_scope("Z_Layer_1"):

                layer_dz = Conv2D(filters=512, kernel_size=1, strides=(1, 1),padding="same",
                                  kernel_initializer=init_kernel)(layer_dz)
                layer_dz = LeakyReLU(alpha=self.config.leakyReLU_alpha)(layer_dz)
                layer_dz = Dropout(rate=self.config.dropout_rate)(layer_dz)

            with tf.variable_scope("Z_Layer_2"):

                layer_dz = Conv2D(filters=512, kernel_size=1, strides=(1, 1),padding="same",
                                  kernel_initializer=init_kernel)(layer_dz)
                layer_dz = LeakyReLU(alpha=self.config.leakyReLU_alpha)(layer_dz)
                layer_dz = Dropout(rate=self.config.dropout_rate)(layer_dz)

            inputs_y = Concatenate(axis=-1)([layer_dx, layer_dz])

            with tf.variable_scope("Y_Layer_1"):

                layer_dy = Conv2D(filters=1024, kernel_size=1, strides=(1, 1), padding="same",
                                  kernel_initializer=init_kernel)(inputs_y)
                layer_dy = LeakyReLU(alpha=self.config.leakyReLU_alpha)(layer_dy)
                layer_dy = Dropout(rate=self.config.dropout_rate)(layer_dy)

            intermediate_layer = layer_dy

            with tf.variable_scope("Y_Layer_2"):
                layer_dy = Conv2D(filters=1024, kernel_size=1, strides=(1, 1), padding="same",
                                  kernel_initializer=init_kernel)(layer_dy)
                layer_dy = Reshape((1024,))(layer_dy)
            self.discriminator_xz = Model(inputs=[inputs_x, inputs_z],outputs=[layer_dy,intermediate_layer])


        # Discriminator model for xx TODO
        # Discriminates between (x, x) and (x, rec_x)
        with tf.variable_scope("Discriminator_xx"):
            input_x = Input(shape=self.config.image_dims, name="input_data")
            rec_x = Input(shape=self.config.image_dims, name="reconstructed_data")

            layer_xx = Concatenate(axis=1)([input_x, rec_x])
            with tf.variable_scope("Layer_1"):
                layer_xx = Conv2D(filters=64, kernel_size=5, strides=(2, 2), padding="same",
                                  kernel_initializer=init_kernel,name="conv1")(layer_xx)
                layer_xx = LeakyReLU(alpha=self.config.leakyReLU_alpha)(layer_xx)
                layer_xx = Dropout(rate=self.config.dropout_rate)(layer_xx)
            # with tf.variable_scope("Layer_1", reuse=True):
            #     weights = tf.get_variable("conv1/kernel")

            with tf.variable_scope("Layer_2"):
                layer_xx = Conv2D(filters=128, kernel_size=5, strides=(2, 2), padding="same",
                                  kernel_initializer=init_kernel, name="conv2")(layer_xx)
                layer_xx = LeakyReLU(alpha=self.config.leakyReLU_alpha)(layer_xx)
                layer_xx = Dropout(rate=self.config.dropout_rate)(layer_xx)
            layer_xx = Flatten()(layer_xx)
            intermediate_layer = layer_xx
            with tf.variable_scope("Layer_3"):
                layer_xx = Dense(units=1, kernel_initializer=init_kernel, name="fc")(layer_xx)

            self.discriminator_xx = Model(inputs=[input_x, rec_x],outputs=[layer_xx,intermediate_layer])


        # Discriminator model for zz TODO
        with tf.variable_scope("Discriminator_zz"):
            input_z = Input(shape=[self.config.noise_dim])
            rec_z = Input(shape=[self.config.noise_dim])
            layer_zz = Concatenate(axis=-1)([input_z,rec_z])

            with tf.variable_scope("Layer_1"):

                layer_zz = Dense(units=64, kernel_initializer=init_kernel)(layer_zz)
                layer_zz = LeakyReLU(alpha=self.config.leakyReLU_alpha)(layer_zz)
                layer_zz = Dropout(rate=self.config.dropout_rate)(layer_zz)

            with tf.variable_scope("Layer_2"):

                layer_zz = Dense(units=32, kernel_initializer=init_kernel)(layer_zz)
                layer_zz = LeakyReLU(alpha=self.config.leakyReLU_alpha)(layer_zz)
                layer_zz = Dropout(rate=self.config.dropout_rate)(layer_zz)
            intermediate_layer = layer_zz
            with tf.variable_scope("Layer_3"):
                layer_zz = Dense(units=1, kernel_initializer=init_kernel)(layer_zz)
                layer_zz = LeakyReLU(alpha=self.config.leakyReLU_alpha)(layer_zz)
                layer_zz = Dropout(rate=self.config.dropout_rate)(layer_zz)

            self.discriminator_zz = Model(inputs=[input_z, rec_z],outputs=[layer_zz, intermediate_layer])


        # Logic of the Graph TODO

        # Losses TODO

        # Optimizers TODO

        # Tensorboard savings TODO

    # Implementatiton of loss functions TODO

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)



