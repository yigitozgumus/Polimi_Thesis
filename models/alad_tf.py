import tensorflow as tf

from base.base_model import BaseModel
import utils.alad_utils as sn

class ALAD_TF(BaseModel):
    def __init__(self, config):
        """
        Args:
            config:
        """
        super(ALAD_TF, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # Placeholders
        self.is_training = tf.placeholder(tf.bool)
        self.init_kernel = tf.random_normal_initializer(mean=0.0, stddev=0.01)

    def encoder(self, img_tensor,getter=None, reuse=False,do_spectral_norm=True):

        """ Encoder architecture in tensorflow
        Maps the image data to the latent space
        Args:
            img_tensor: input data for the encoder
            getter: for exponential moving average during inference
            reuse: sharing variables or not
            do_spectral_norm:
        """
        # Change the layer type if do_spectral_norm is true
        layers = sn if do_spectral_norm else tf.layers
        with tf.variable_scope("Encoder", reuse=reuse, custom_getter=getter) as scope:
            img_tensor = tf.reshape(img_tensor, [-1,28,28,1])
            net_name = "layer_1"
            with tf.variable_scope(net_name):
                net = layers.conv2d(img_tensor, filters=128, kernel_size=4, strides=(2, 2),padding="same",
                                    kernel_initializer=self.init_kernel)
                net = tf.layers.BatchNormalization()





    def generator(self, noise_tensor, getter=None, reuse=False):
        """ Generator architecture in tensorflow

        Generates the data from the latent space
        Args:
            noise_tensor: input variable in the latent space
            getter: for exponential moving average during inference
            reuse: sharing variables or not
        """
        pass

    def discriminator_xz(self, img_tensor, noise_tensor, getter=None, reuse=False,
                         do_spectral_norm=True):
        """ Discriminator architecture in tensorflow

        Discriminates between pairs (E(x), x) and (z, G(z))
        Args:
            img_tensor:
            noise_tensor:
            getter: for exponential moving average during inference
            reuse: sharing variables or not
            do_spectral_norm:
        """
        pass

    def discriminator_xx(self,img_tensor, recreated_img, getter=None, reuse=False,
                         do_spectral_norm=True):
        """ Discriminator architecture in tensorflow

        Discriminates between (x, x) and (x, rec_x)
        Args:
            img_tensor:
            recreated_img:
            getter: for exponential moving average during inference
            reuse: sharing variables or not
            do_spectral_norm:
        """
        pass

    def discriminator_zz(self,noise_tensor, recreated_noise, getter=None, reuse=False,
                         do_spectral_norm=True):
        """ Discriminator architecture in tensorflow

        Discriminates between (z, z) and (z, rec_z)
        Args:
            noise_tensor:
            recreated_noise:
            getter: for exponential moving average during inference
            reuse: sharing variables or not
            do_spectral_norm:
        """
        pass

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
