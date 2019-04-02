
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class BaseTrainKeras:
    def __init__(self,sess, model,data, config):
        self.model = model
        self.config = config
        self.data = data
        self.sess = sess
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)
        self.rows = int(np.sqrt(self.config.num_example_imgs_to_generate))


    def train(self):
        raise NotImplementedError


    def save_generated_images(self,predictions, epoch):
        # make sure the training parameter is set to False because we
        # don't want to train the batchnorm layer when doing inference.
        predictions = np.asarray(predictions)[0]
        fig = plt.figure(figsize=(self.rows, self.rows))
        for i in range(predictions.shape[0]):
            plt.subplot(self.rows, self.rows, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(self.config.step_generation_dir + 'image_at_epoch_{:04d}.png'.format(epoch))
        plt.close()

