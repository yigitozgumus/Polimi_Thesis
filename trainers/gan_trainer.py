
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
        # Attach the epoch loop to a variable
        loop = tqdm(range(self.config.num_iter_per_epoch))
        # Define the lists for summaries and losses
        gen_losses = []
        disc_losses = []
        summaries = []
        # Get the current epoch counter
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        # Make the iterator
        #iterator = self.data.make_initializable_iterator()
        # initialize the image batch
        #next_element = iterator.get_next()
        self.sess.run(self.data.iterator.initializer)
        for epoch in loop:
            # Calculate the losses and obtain the summaries to write
            gen_loss, disc_loss,summary = self.train_step(self.data.next_element)
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)
            summaries.append(summary)
        # write the summaries
        self.logger.summarize(cur_epoch, summaries=summaries)
        # Compute the means of the losses
        gen_loss = np.mean(gen_losses)
        disc_loss = np.mean(disc_losses)
        # Generate images between epochs to evaluate   
        rand_noise = self.sess.run(self.model.random_vector_for_generation)
        feed_dict = {self.model.noise_input: rand_noise}
        generator_predictions = self.sess.run(
                [self.model.progress_images], feed_dict=feed_dict)
        self.save_generated_images(generator_predictions, cur_epoch)

        if (cur_epoch % self.config.show_steps == 0 or cur_epoch == 1):
                print('Epoch {}: Generator Loss: {}, Discriminator Loss: {}'.format(
                    cur_epoch+1, gen_loss, disc_loss))
  
        self.model.save(self.sess)

    #@tf.contrib.eager.defun
    def train_step(self,image):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        # Generate noise from uniform  distribution between -1 and 1
        noise = np.random.uniform(-1.,1.,size=[self.config.batch_size,self.config.noise_dim])
        image_eval = self.sess.run(image)
        feed_dict = {self.model.noise_input: noise, self.model.real_image_input: image_eval}
        gen_loss, disc_loss,_,_,summary = self.sess.run(
            [self.model.gen_loss, self.model.total_disc_loss, self.model.train_gen, self.model.train_disc,self.model.summary], feed_dict=feed_dict)

        return gen_loss, disc_loss, summary,

        


