import tensorflow as tf
from base.base_model import BaseModel



class GAN(BaseModel):

    def __init__(self, config):
        super(GAN, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):


        # Make the Generator model
        with tf.name_scope("Generator"):
            inputs_g = tf.keras.layers.Input(shape=[self.config.noise_dim])
            x = tf.keras.layers.Dense(7*7*256, use_bias=False)(inputs_g)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU()(x)
            x = tf.keras.layers.Reshape((7, 7, 256))(x)

            assert x.get_shape().as_list() == [None, 7, 7, 256]
            x = tf.keras.layers.Conv2DTranspose(
                128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
            assert x.get_shape().as_list() == [None, 7, 7, 128]
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU()(x)

            x = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
            assert x.get_shape().as_list() == [None, 14, 14, 64]
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU()(x)

            x = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(
                2, 2), padding='same', use_bias=False, activation='tanh')(x)
            assert x.get_shape().as_list() == [None, 28, 28, 1]
            self.generator = tf.keras.models.Model(inputs=inputs_g, outputs=x)

        # Make the discriminator model
        with tf.name_scope("Discriminator"):
            inputs_d = tf.keras.layers.Input(shape=self.config.state_size)
            x = tf.keras.layers.Conv2D(32, (5,5),strides=(2, 2),padding='same')(inputs_d)
            x = tf.keras.layers.LeakyReLU()(x)
            #x = tf.keras.layers.AveragePooling2D(pool_size=(2 ,2),padding='same')(x)
            x = tf.keras.layers.Dropout(rate=0.3)(x)
            x = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
            x = tf.keras.layers.LeakyReLU()(x)
            x = tf.keras.layers.Dropout(rate=0.3)(x)

            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(1)(x)
            self.discriminator = tf.keras.models.Model(inputs=inputs_d,outputs=x)

        # Initialization of Optimizers
        self.generator_optimizer = tf.train.AdamOptimizer(self.config.optimizer_learning_rate)
        self.discriminator_optimizer = tf.train.AdamOptimizer(self.config.optimizer_learning_rate)

    # Implementation of losses
    def generator_loss(self,generated_output):
        return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)

    def discriminator_loss(self,real_output, generated_output):
        real_loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=tf.ones_like(real_output), logits=real_output)

        generated_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

        total_loss = real_loss + generated_loss

        return total_loss


    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)



