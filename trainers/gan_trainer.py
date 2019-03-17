from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import time

class GANTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
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
        for epoch in loop:
            start = time.time()
            gen_loss, disc_loss = self.train_step()
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)
        gen_loss = tf.reduce_mean(gen_losses)
        disc_loss = tf.reduce_mean(disc_losses)
        if (self.model.cur_epoch_tensor % self.config.show_steps == 0 or self.model.cur_epoch_tensor == 1):
                print('Epoch {}: Generator Loss: {}, Discriminator Loss: {}'.format(
                    self.model.cur_epoch_tensor, gen_loss, disc_loss))
        
        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            "gen_loss": gen_loss,
            "disc_loss": disc_loss,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)


    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - return any metrics you need to summarize
       """
       # Generate noise from normal distribution
        noise = tf.random_normal([self.config.batch_size,self.config.noise_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.model.generator(noise, training=True)
            real_output = self.model.discriminator(self.data, training=True)

            gen_loss = self.model.generator_loss(generated_output=generated_images)
            disc_loss = self.model.discriminator_loss(real_output=real_output,generated_output=generated_images)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.model.generator.variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.model.discriminator.variables)

        self.model.generator_optimizer.apply_gradients(zip(gradients_of_generator,self.model.generator.variables))
        self.model.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,self.model.discriminator.variables))

        return gen_loss, disc_loss


        


