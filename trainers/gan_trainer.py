from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf
#tf.enable_eager_execution()
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
        img_gens = []
        img_discs = []
        for _ in loop:
            img_gen,img_disc, gen_loss, disc_loss = self.train_step()
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)
            img_gens.append(img_gen)
            img_discs.append(img_disc)
            
        gen_loss = tf.math.reduce_mean(gen_losses)
        disc_loss = tf.math.reduce_mean(disc_losses)
        
        tf.summary.scalar("Generator_Loss", gen_loss)
        tf.summary.scalar("Discriminator_Loss", disc_loss)
        
        #x_image = tf.summary.image('FromNoise', tf.reshape(img_gens[-1], [-1, 28, 28, 1]), 4)
        #x_image2 = tf.summary.image('FromGen', tf.reshape(img_disc[-1], [-1, 28, 28, 1]), 4)
        
#         current = self.sess.run(self.model.cur_epoch_tensor)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        if (cur_it % self.config.show_steps == 0 or cur_it == 1):
                print('Epoch {}: Generator Loss: {}, Discriminator Loss: {}'.format(
                    cur_it, gen_loss, disc_loss))
        
#         summaries_dict = {
#             "g_loss": gen_loss,
#             "d_loss": disc_loss,
#         }
        summary = tf.summary.merge_all()

        self.logger.summarize(cur_it, summaries=sum)
        self.model.save(self.sess)

    #@tf.contrib.eager.defun
    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - return any metrics you need to summarize
       """
       # Generate noise from normal distribution
        noise = tf.random_normal([self.config.batch_size,self.config.noise_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_image = self.model.generator(noise, training=True)
            
            real_output = self.model.discriminator(self.data, training=True)
            generated_output = self.model.discriminator(generated_image, training=True)
            with tf.name_scope('Generator_Loss'):
                gen_loss = self.model.generator_loss(generated_output)
            with tf.name_scope('Discriminator_Loss'):    
                disc_loss = self.model.discriminator_loss(real_output,generated_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.model.generator.variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.model.discriminator.variables)

        self.model.generator_optimizer.apply_gradients(zip(gradients_of_generator,self.model.generator.variables))
        self.model.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,self.model.discriminator.variables))

        return generated_image,generated_output,gen_loss, disc_loss


        


