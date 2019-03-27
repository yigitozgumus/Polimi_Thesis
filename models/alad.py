
from base.base_model import BaseModel
import tensorflow as tf


class ALAD(BaseModel):
    def __init__(self, config):
        super(ALAD, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        # Placeholders for the data
        # Noise placeholder [None, 100]
        self.noise_input = tf.placeholder(
            tf.float32,shape=[None, self.config.noise_dim]
        )
        # Real Image place holder [None, 28, 28, 1]
        self.real_image_output = tf.placeholder(
            tf.float32, shape=[None] + self.config.image_dims
        )
        # Encoder Model TODO

        # Generator Model TODO

        # Discriminator model for xz TODO

        # Discriminator model for xx TODO

        # Discriminator model for zz TODO

        # Logic of the Graph TODO

        # Losses TODO

        # Optimizers TODO

        # Tensorboard savings TODO

    # Implementatiton of loss functions TODO

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
    
    

