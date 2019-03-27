from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf

# tf.enable_eager_execution()
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
        # Attach the epoch loop to a variable
        loop = tqdm(range(self.config.num_iter_per_epoch))
        # Define the lists for summaries and losses
        gen_losses = []
        disc_losses = []
        summaries = []
        true_accuracies = []
        fake_accuracies = []
        tot_accuracies = []
        # Get the current epoch counter
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        self.sess.run(self.data.iterator.initializer)
        for epoch in loop:
            # Calculate the losses and obtain the summaries to write
            gen_loss, disc_loss, fake_acc, true_acc, tot_acc, summary = self.train_step(
                self.data.next_element,
                cur_epoch=cur_epoch
            )
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)
            fake_accuracies.append(fake_acc)
            true_accuracies.append(true_acc)
            tot_accuracies.append(tot_acc)
            summaries.append(summary)
        # write the summaries
        self.logger.summarize(cur_epoch, summaries=summaries)
        # Compute the means of the losses
        gen_loss_m = np.mean(gen_losses)
        disc_loss_m = np.mean(disc_losses)
        fake_acc_m = np.mean(fake_accuracies)
        true_acc_m = np.mean(true_accuracies)
        total_acc_m = np.mean(tot_accuracies)
        # Generate images between epochs to evaluate
        rand_noise = self.sess.run(self.model.random_vector_for_generation)
        feed_dict = {self.model.noise_tensor: rand_noise}
        generator_predictions = self.sess.run(
            [self.model.progress_images], feed_dict=feed_dict
        )
        self.save_generated_images(generator_predictions, cur_epoch)

        if cur_epoch % self.config.show_steps == 0 or cur_epoch == 1:
            print(
                "Epoch {}: Generator Loss: {}, Discriminator Loss: {},\
                    Fake Accuracy: {}, True Accuracy: {}, Total Accuracy: {}".format(
                    cur_epoch + 1, gen_loss_m, disc_loss_m, fake_acc_m, true_acc_m, total_acc_m
                )
            )

        self.model.save(self.sess)

    # @tf.contrib.eager.defun
    def train_step(self, image,cur_epoch):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        # Generate noise from uniform  distribution between -1 and 1
        # New Noise Generation
        # noise = np.random.uniform(-1., 1.,size=[self.config.batch_size, self.config.noise_dim])
        sigma = max(0.75*(10. - cur_epoch) / (10), 0.05)
        noise = np.random.normal(
            loc=0.0, scale=1.0, size=[self.config.batch_size, self.config.noise_dim]
        )
        real_noise = np.random.normal(
            scale=sigma, size=[self.config.batch_size] + self.config.image_dims
        )
        fake_noise = np.random.normal(
            scale=sigma, size=[self.config.batch_size] + self.config.image_dims
        )
        image_eval = self.sess.run(image)
        feed_dict = {
            self.model.noise_tensor: noise,
            self.model.image_input: image_eval,
            self.model.real_noise: real_noise,
            self.model.fake_noise: fake_noise,
        }

        gen_loss, disc_loss, fake_acc, true_acc, tot_acc, _, _, summary = self.sess.run(
            [ 
                self.model.gen_loss,
                self.model.total_disc_loss,
                self.model.accuracy_fake,
                self.model.accuracy_real,
                self.model.accuracy_total,
                self.model.train_gen,
                self.model.train_disc,
                self.model.summary,
            ],
            feed_dict=feed_dict,
        )

        return gen_loss, disc_loss, fake_acc, true_acc, tot_acc, summary
