
from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf
#tf.enable_eager_execution()
import time

class GANTrainer(BaseTrain):
    def __init__(self, sess, model,data, config, logger):
        super(GANTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        loop = tqdm(range(self.config.num_iter_per_epoch))
        gen_losses = []
        disc_losses = []
        summaries = []
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        # Make the iterator initializor
        iterator = self.data.make_initializable_iterator()
        next_element = iterator.get_next()
        self.sess.run(iterator.initializer)
        for epoch in loop:
            gen_loss, disc_loss,summary = self.train_step(next_element)
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)
            summaries.append(summary)
        self.logger.summarize(cur_epoch, summaries=summaries)
        gen_loss = tf.math.reduce_mean(gen_losses).eval(session=self.sess)
        disc_loss = tf.math.reduce_mean(disc_losses).eval(session=self.sess)
        random_vector_for_generation = tf.random_normal(
            [self.config.num_example_imgs_to_generate, self.config.noise_dim])
        if (cur_epoch % self.config.num_epochs_to_test == 0 or cur_epoch == 1):
            rand_noise = self.sess.run(random_vector_for_generation)
            feed_dict = {self.model.noise_input: rand_noise}
            generator_predictions = self.sess.run(
                [self.model.progress_images], feed_dict=feed_dict)
            self.save_generated_images(generator_predictions, cur_epoch)
        if (cur_epoch % self.config.show_steps == 0 or cur_epoch == 1):
                print('Epoch {}: Generator Loss: {}, Discriminator Loss: {}'.format(
                    cur_epoch, gen_loss, disc_loss))

        
        self.model.save(self.sess)

    #@tf.contrib.eager.defun
    def train_step(self,image):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        # Generate noise from normal distribution
        noise = tf.random_normal([self.config.batch_size,self.config.noise_dim])
        noise_gen = self.sess.run(noise)
        image_eval = self.sess.run(image)
        feed_dict = {self.model.noise_input: noise_gen, self.model.real_image_input: image_eval}
        gen_loss, disc_loss,_,_,summary = self.sess.run(
            [self.model.gen_loss, self.model.disc_loss, self.model.train_gen, self.model.train_disc,self.model.summary], feed_dict=feed_dict)

        return gen_loss, disc_loss, summary,

        


