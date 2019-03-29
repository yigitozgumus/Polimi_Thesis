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
                "Epoch {} --\nGenerator Loss: {}\nDiscriminator Loss: {}\nFake Accuracy: {}\nTrue Accuracy: {}\nTotal Accuracy: {}".format(
                    cur_epoch + 1, gen_loss_m, disc_loss_m, fake_acc_m, true_acc_m, total_acc_m
                )
            )

        self.model.save(self.sess)

    def train_step(self, image, cur_epoch):

        # Generate noise from uniform  distribution between -1 and 1
        # New Noise Generation
        # noise = np.random.uniform(-1., 1.,size=[self.config.batch_size, self.config.noise_dim])
        noise_probability = self.config.noise_probability
        sigma = max(0.75 * (10. - cur_epoch) / (10), 0.05)
        noise = np.random.normal(loc=0.0, scale=1.0, size=[self.config.batch_size, self.config.noise_dim])
        true_labels = np.zeros((self.config.batch_size,1)) + np.random.uniform(low=0.0, high=0.1, size=[self.config.batch_size, 1])
        flipped_idx = np.random.choice(np.arange(len(true_labels)), size= int(noise_probability * len(true_labels)))
        true_labels[flipped_idx] = 1 - true_labels[flipped_idx]
        generated_labels = np.ones((self.config.batch_size,1)) - np.random.uniform(low=0.0, high=0.1, size=[self.config.batch_size, 1])
        flipped_idx = np.random.choice(np.arange(len(generated_labels)), size=int(noise_probability * len(generated_labels)))
        generated_labels[flipped_idx] =  1 - generated_labels[flipped_idx]
        # Instance noise additions
        if self.config.include_noise:
            # If we want to add this is will add the noises
            real_noise = np.random.normal(scale=sigma, size=[self.config.batch_size] + self.config.image_dims)
            fake_noise = np.random.normal(scale=sigma, size=[self.config.batch_size] + self.config.image_dims)
        else:
            # Otherwise we are just going to add zeros which will not break anything
            real_noise = np.zeros(([self.config.batch_size] + self.config.image_dims))
            fake_noise = np.zeros(([self.config.batch_size] + self.config.image_dims))
        # Evaluation of the image
        image_eval = self.sess.run(image)
        # Construct the Feed Dictionary
        # Train the Discriminator on both real and fake images
        disc_loss, fake_acc, true_acc, tot_acc, _ = self.sess.run(
            [
                self.model.total_disc_loss,
                self.model.accuracy_fake,
                self.model.accuracy_real,
                self.model.accuracy_total,
                self.model.train_disc
            ],
            feed_dict={
                self.model.noise_tensor: noise,
                self.model.image_input: image_eval,
                self.model.real_noise: real_noise,
                self.model.fake_noise: fake_noise,
                self.model.true_labels: true_labels,
                self.model.generated_labels: generated_labels
            }
        )
        # Train the Generator and get the summaries
        # Re create the noise for the generator
        noise = np.random.normal(loc=0.0, scale=1.0, size=[self.config.batch_size, self.config.noise_dim])
        gen_loss, _,summary = self.sess.run(
            [self.model.gen_loss,
             self.model.train_gen,
             self.model.summary],
            feed_dict={
                self.model.noise_tensor: noise,
                self.model.image_input: image_eval,
                self.model.real_noise: real_noise,
                self.model.fake_noise: fake_noise,
                self.model.true_labels: true_labels,
                self.model.generated_labels: generated_labels
            }
            )
        # Retrain the Generator
        gen_loss, _, summary = self.sess.run(
            [self.model.gen_loss,
             self.model.train_gen,
             self.model.summary],
            feed_dict={
                self.model.noise_tensor: noise,
                self.model.image_input: image_eval,
                self.model.real_noise: real_noise,
                self.model.fake_noise: fake_noise,
                self.model.true_labels: true_labels,
                self.model.generated_labels: generated_labels
            }
        )


        return gen_loss, disc_loss, fake_acc, true_acc, tot_acc, summary
