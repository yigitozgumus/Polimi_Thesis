
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
        # Placeholders for the noise and the image
        self.noise_input = tf.placeholder(tf.float32, shape=[None,self.config.noise_dim])
        self.real_image_input = tf.placeholder(tf.float32, shape=[None] + self.config.image_dims)

        # Make the Generator Model
        with tf.name_scope("Generator"):
            # Keras Model Object is Defined
            model_g = Sequential()
            model_g.add(Dense(128 * 7 * 7, activation="relu", input_shape=self.config.noise_dim))
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

            model_g.summary()
            input_g = Input(shape=[self.config.noise_dim])
            output_img_g = model_g(input_g)

            self.generator = Model(input_g, output_img_g)

        # Make the Discriminator Model
        with tf.name_scope("Discriminator"):
            # Keras Model object is defined
            model_d = Sequential()
            model_d.add(Conv2D(32, kernel_size=3, strides=(2, 2),
                             input_shape=self.config.state_size, padding="same"))
            model_d.add(LeakyReLU(alpha=self.config.leakyReLU_alpha))
            model_d.add(Dropout(self.config.dropout_rate))
            model_d.add(Conv2D(64, kernel_size=3, strides=(2, 2), padding="same"))
            model_d.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
            model_d.add(BatchNormalization(momentum=self.config.batch_momentum))
            model_d.add(LeakyReLU(alpha=self.config.leakyReLU_alpha))
            model_d.add(Dropout(self.config.dropout_rate))
            model_d.add(Conv2D(128, kernel_size=3, strides=(2, 2), padding="same"))
            model_d.add(BatchNormalization(momentum=0.8))
            model_d.add(LeakyReLU(alpha=self.config.leakyReLU_alpha))
            model_d.add(Dropout(self.config.dropout_rate))
            model_d.add(Conv2D(256, kernel_size=3, strides=(1, 1), padding="same"))
            model_d.add(BatchNormalization(momentum=0.8))
            model_d.add(LeakyReLU(alpha=self.config.leakyReLU_alpha))
            model_d.add(Dropout(self.config.dropout_rate))
            model_d.add(Flatten())
            model_d.add(Dense(1, activation='sigmoid'))

            model_d.summary()
            input_d = Input(shape=self.config.state_size)
            output_validity = model_d(input_d)

            self.discriminator = Model(input_d, output_validity)

            
            



    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

