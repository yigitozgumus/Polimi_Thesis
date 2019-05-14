from base.base_model_keras import BaseModelKeras
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    Reshape,
)
from tensorflow.keras.layers import Conv2DTranspose, ReLU, Dropout
from tensorflow.keras.layers import Concatenate, Flatten, Dense
from tensorflow.keras.models import Model


class ALADKeras(BaseModelKeras):
    def __init__(self, config):
        super(ALADKeras, self).__init__(config)

        self.build_model()

    def build_model(self):
        # Kernel Initialization
        init_kernel = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        # Encoder Model TODO
        # Maps the data into the Latent Space

        ##################################
        # ENCODER
        ##################################
        inputs_e = Input(shape=self.config.trainer.image_dims)
        layer_e = Conv2D(
            filters=128,
            kernel_size=4,
            strides=(2, 2),
            padding="same",
            kernel_initializer=init_kernel,
        )(inputs_e)
        layer_e = BatchNormalization(momentum=self.config.trainer.batch_momentum)(
            layer_e
        )
        layer_e = LeakyReLU(alpha=self.config.trainer.leakyReLU_alpha)(layer_e)

        layer_e = Conv2D(
            filters=256,
            kernel_size=4,
            strides=(2, 2),
            padding="same",
            kernel_initializer=init_kernel,
        )(layer_e)
        layer_e = BatchNormalization(momentum=self.config.trainer.batch_momentum)(
            layer_e
        )
        layer_e = LeakyReLU(alpha=self.config.trainer.leakyReLU_alpha)(layer_e)

        layer_e = Conv2D(
            filters=512,
            kernel_size=4,
            strides=(2, 2),
            padding="same",
            kernel_initializer=init_kernel,
        )(layer_e)
        layer_e = BatchNormalization(momentum=self.config.trainer.batch_momentum)(
            layer_e
        )
        layer_e = LeakyReLU(alpha=self.config.trainer.leakyReLU_alpha)(layer_e)

        layer_e = Conv2D(
            filters=self.config.trainer.noise_dim,
            kernel_size=4,
            strides=(1, 1),
            padding="valid",
            kernel_initializer=init_kernel,
        )(layer_e)
        layer_e = Reshape((self.config.trainer.noise_dim,))(layer_e)
        self.encoder = Model(inputs=inputs_e, outputs=layer_e)

        ##################################
        # GENERATOR
        ##################################
        # Generate data from the latent space

        inputs_g = Input(shape=[self.config.trainer.noise_dim])
        layer_g = Dense(units=7 * 7 * 512, kernel_initializer=init_kernel, name="fc")(
            inputs_g
        )
        layer_g = Reshape((7, 7, 512))(layer_g)

        # layer_g = Reshape((1, 1, self.config.trainer.noise_dim))(inputs_g)

        layer_g = Conv2DTranspose(
            filters=512,
            kernel_size=5,
            strides=(1, 1),
            padding="same",
            kernel_initializer=init_kernel,
        )(layer_g)
        layer_g = BatchNormalization(momentum=self.config.trainer.batch_momentum)(
            layer_g
        )
        layer_g = ReLU()(layer_g)

        layer_g = Conv2DTranspose(
            filters=256,
            kernel_size=5,
            strides=(2, 2),
            padding="same",
            kernel_initializer=init_kernel,
        )(layer_g)
        layer_g = BatchNormalization(momentum=self.config.trainer.batch_momentum)(
            layer_g
        )
        layer_g = ReLU()(layer_g)

        layer_g = Conv2DTranspose(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
            padding="same",
            kernel_initializer=init_kernel,
        )(layer_g)
        layer_g = BatchNormalization(momentum=self.config.trainer.batch_momentum)(
            layer_g
        )
        layer_g = ReLU()(layer_g)
        layer_g = Conv2DTranspose(
            filters=1,
            kernel_size=5,
            strides=(1, 1),
            padding="same",
            activation="tanh",
            kernel_initializer=init_kernel,
        )(layer_g)
        self.generator = Model(inputs=inputs_g, outputs=layer_g)

        ##################################
        # DISCRIMINATOR XZ
        ##################################
        # Discriminates between pairs (E(x), x) and (G(z), z)

        inputs_x = Input(shape=self.config.trainer.image_dims, name="image_input")
        inputs_z = Input(shape=[self.config.trainer.noise_dim], name="noise_input")

        layer_dx = Conv2D(
            filters=128,
            kernel_size=4,
            strides=(2, 2),
            padding="same",
            kernel_initializer=init_kernel,
        )(inputs_x)
        layer_dx = LeakyReLU(alpha=self.config.trainer.leakyReLU_alpha)(layer_dx)

        layer_dx = Conv2D(
            filters=256,
            kernel_size=4,
            strides=(2, 2),
            padding="same",
            kernel_initializer=init_kernel,
        )(layer_dx)
        layer_dx = BatchNormalization(momentum=self.config.trainer.batch_momentum)(
            layer_dx
        )
        layer_dx = LeakyReLU(alpha=self.config.trainer.leakyReLU_alpha)(layer_dx)

        layer_dx = Conv2D(
            filters=512,
            kernel_size=4,
            strides=(2, 2),
            padding="same",
            kernel_initializer=init_kernel,
        )(layer_dx)
        layer_dx = BatchNormalization(momentum=self.config.trainer.batch_momentum)(
            layer_dx
        )
        layer_dx = LeakyReLU(alpha=self.config.trainer.leakyReLU_alpha)(layer_dx)

        layer_dx = Reshape((1, 1, 512 * 14 * 14))(layer_dx)
        #
        # layer_dz = Reshape((1, 1, self.config.trainer.noise_dim))(inputs_z)
        #
        # layer_dz = Conv2D(filters=512, kernel_size=1, strides=(1, 1), padding="same",
        #                   kernel_initializer=init_kernel)(layer_dz)
        # layer_dz = LeakyReLU(alpha=self.config.trainer.leakyReLU_alpha)(layer_dz)
        # layer_dz = Dropout(rate=self.config.trainer.dropout_rate)(layer_dz)
        #
        # layer_dz = Conv2D(filters=512, kernel_size=1, strides=(1, 1), padding="same",
        #                   kernel_initializer=init_kernel)(layer_dz)
        # layer_dz = LeakyReLU(alpha=self.config.trainer.leakyReLU_alpha)(layer_dz)
        # layer_dz = Dropout(rate=self.config.trainer.dropout_rate)(layer_dz)
        #
        # inputs_y = Concatenate(axis=-1)([layer_dx, layer_dz])
        #
        # layer_dy = Conv2D(filters=1024, kernel_size=1, strides=(1, 1), padding="same",
        #                   kernel_initializer=init_kernel)(inputs_y)
        # layer_dy = LeakyReLU(alpha=self.config.trainer.leakyReLU_alpha)(layer_dy)
        # layer_dy = Dropout(rate=self.config.trainer.dropout_rate)(layer_dy)
        #
        # intermediate_layer = layer_dy
        # layer_dy = Conv2D(filters=1024, kernel_size=1, strides=(1, 1), padding="same",
        #                   kernel_initializer=init_kernel)(layer_dy)
        # layer_dy = Reshape((1024,))(layer_dy)
        self.discriminator_xz = Model(inputs=[inputs_x, inputs_z], outputs=[layer_dx])

        ##################################
        # DISCRIMINATOR XX
        ##################################
        # Discriminates between (x, x) and (x, rec_x)
        input_x = Input(shape=self.config.trainer.image_dims, name="input_data")
        rec_x = Input(shape=self.config.trainer.image_dims, name="reconstructed_data")

        layer_xx = Concatenate(axis=1)([input_x, rec_x])
        layer_xx = Conv2D(
            filters=64,
            kernel_size=5,
            strides=(2, 2),
            padding="same",
            kernel_initializer=init_kernel,
            name="conv1",
        )(layer_xx)
        layer_xx = LeakyReLU(alpha=self.config.trainer.leakyReLU_alpha)(layer_xx)
        layer_xx = Dropout(rate=self.config.trainer.dropout_rate)(layer_xx)

        layer_xx = Conv2D(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
            padding="same",
            kernel_initializer=init_kernel,
            name="conv2",
        )(layer_xx)
        layer_xx = LeakyReLU(alpha=self.config.trainer.leakyReLU_alpha)(layer_xx)
        layer_xx = Dropout(rate=self.config.trainer.dropout_rate)(layer_xx)
        layer_xx = Flatten()(layer_xx)
        intermediate_layer = layer_xx
        layer_xx = Dense(units=1, kernel_initializer=init_kernel, name="fc")(layer_xx)

        self.discriminator_xx = Model(
            inputs=[input_x, rec_x], outputs=[layer_xx, intermediate_layer]
        )

        ##################################
        # DISCRIMINATOR ZZ
        ##################################
        # Discriminator model for zz TODO

        input_z = Input(shape=[self.config.trainer.noise_dim])
        rec_z = Input(shape=[self.config.trainer.noise_dim])
        layer_zz = Concatenate(axis=-1)([input_z, rec_z])

        layer_zz = Dense(units=64, kernel_initializer=init_kernel)(layer_zz)
        layer_zz = LeakyReLU(alpha=self.config.trainer.leakyReLU_alpha)(layer_zz)
        layer_zz = Dropout(rate=self.config.trainer.dropout_rate)(layer_zz)

        layer_zz = Dense(units=32, kernel_initializer=init_kernel)(layer_zz)
        layer_zz = LeakyReLU(alpha=self.config.trainer.leakyReLU_alpha)(layer_zz)
        layer_zz = Dropout(rate=self.config.trainer.dropout_rate)(layer_zz)
        intermediate_layer = layer_zz
        layer_zz = Dense(units=1, kernel_initializer=init_kernel)(layer_zz)
        layer_zz = LeakyReLU(alpha=self.config.trainer.leakyReLU_alpha)(layer_zz)
        layer_zz = Dropout(rate=self.config.trainer.dropout_rate)(layer_zz)

        self.discriminator_zz = Model(
            inputs=[input_z, rec_z], outputs=[layer_zz, intermediate_layer]
        )

        # Logic of the Graph TODO

        # Losses TODO

        # Optimizers TODO

        # Tensorboard savings TODO

    # Implementatiton of loss functions TODO

    def save(self, checkpoint_path):
        pass

    def load(self, checkpoint_path):
        pass
