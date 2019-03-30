from base.base_train_eager import BaseTrain_eager
import tensorflow as tf
from tqdm import tqdm
import numpy as np


class GANTrainer_eager(BaseTrain_eager):
    def __init__(self, model, data, config, logger):
        super(GANTrainer_eager, self).__init__( model, data, config, logger)

    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        # Define the lists for summaries and losses
        gen_losses = []
        total_disc_losses = []
        real_disc_losses = []
        fake_disc_losses = []

        # Get the current epoch
        cur_epoch = self.model.cur_epoch_tensor.numpy()
        for image in self.data.dataset:
            gen_loss, total_dl, real_dl, fake_dl, = self.train_step(image, cur_epoch=cur_epoch)
            gen_losses.append(gen_loss)
            total_disc_losses.append(total_dl)
            real_disc_losses.append(real_dl)
            fake_disc_losses.append(fake_dl)
        # write the summaries
        #self.logger.summarize(cur_epoch, summaries=summaries)
        # Compute the means of the losses
        gen_loss_m = np.mean(gen_losses)
        disc_loss_t = np.mean(total_disc_losses)
        disc_loss_r = np.mean(real_disc_losses)
        disc_loss_f = np.mean(fake_disc_losses)
        with tf.contrib.summary.record_summaries_every_n_global_steps(1):
            tf.contrib.summary.scalar("Generator_Loss", gen_loss_m)
            tf.contrib.summary.scalar("Total_Discriminator_Loss", disc_loss_t)
            tf.contrib.summary.scalar("Real_Discriminator_Loss", disc_loss_r)
            tf.contrib.summary.scalar("Fake_Discriminator_Loss", disc_loss_f)
        # Generate images between epochs to evaluate
        rand_noise = np.random.normal(
            loc=0.0, scale=1.0,
            size=[self.config.num_example_imgs_to_generate, self.config.noise_dim])
        progress_images = self.model.generator(rand_noise, training=False)
        self.save_generated_images(progress_images, cur_epoch)

        if cur_epoch % self.config.show_steps == 0 or cur_epoch == 1:
            print(
                "Epoch {} --\nGenerator Loss: {}\nDiscriminator Loss: {}\n".format(
                    cur_epoch , gen_loss_m, disc_loss_t,
                )
            )
        self.model.save()

    @tf.contrib.eager.defun
    def train_step(self, image, cur_epoch):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        # Generate noise from uniform  distribution between -1 and 1
        noise_probability = self.config.noise_probability
        sigma = max(0.75 * (10. - cur_epoch) / (10), 0.05)
        noise = np.random.normal(loc=0.0, scale=1.0, size=[self.config.batch_size, self.config.noise_dim])
        # Soft Label Generation
        true_labels = np.zeros((self.config.batch_size, 1)) + np.random.uniform(low=0.0, high=0.1,
                                                                size=[self.config.batch_size, 1])
        flipped_idx = np.random.choice(np.arange(len(true_labels)),
                                       size=int(noise_probability * len(true_labels)))
        true_labels[flipped_idx] = 1 - true_labels[flipped_idx]

        generated_labels = np.ones((self.config.batch_size, 1)) - np.random.uniform(
            low=0.0, high=0.1,size=[self.config.batch_size, 1])
        flipped_idx = np.random.choice(np.arange(len(generated_labels)),
                                       size=int(noise_probability * len(generated_labels)))
        generated_labels[flipped_idx] = 1 - generated_labels[flipped_idx]
        # Noise Inclusion
        if self.config.include_noise:
            # If we want to add this is will add the noises
            real_noise = np.random.normal(scale=sigma, size=[self.config.batch_size] + self.config.image_dims)
            fake_noise = np.random.normal(scale=sigma, size=[self.config.batch_size] + self.config.image_dims)
        else:
            # Otherwise we are just going to add zeros which will not break anything
            real_noise = np.zeros(([self.config.batch_size] + self.config.image_dims))
            fake_noise = np.zeros(([self.config.batch_size] + self.config.image_dims))

        # Train the Discriminator
        with tf.GradientTape() as disc_tape:
            real_image = image + real_noise
            generated_sample = self.model.generator(noise, training=True) + fake_noise
            disc_real = self.model.discriminator(real_image, training=True)
            disc_fake = self.model.discriminator(generated_sample, training=True)
            total_disc_loss, real_disc_loss, fake_disc_loss = self.model.discriminator_loss(
                true_labels,generated_labels,disc_real,disc_fake)
        discriminator_gradients = disc_tape.gradient(total_disc_loss,
                                                     self.model.discriminator.variables)
        self.model.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    self.model.discriminator.variables))

        # Train the Generator
        if self.config.include_noise:
            # If we want to add this is will add the noises
            fake_noise = np.random.normal(scale=sigma, size=[self.config.batch_size] + self.config.image_dims)
        else:
            # Otherwise we are just going to add zeros which will not break anything
            fake_noise = np.zeros(([self.config.batch_size] + self.config.image_dims))

        noise = np.random.normal(loc=0.0, scale=1.0, size=[self.config.batch_size, self.config.noise_dim])
        with tf.GradientTape() as gen_tape:
            generated_sample = self.model.generator(noise, training=True) + fake_noise
            stacked_gan = self.model.discriminator(generated_sample,training=True)
            gen_loss = self.model.generator_loss(stacked_gan)
        generator_gradients = gen_tape.gradient(gen_loss, self.model.generator.variables)
        self.model.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                           self.model.generator.variables))

        return gen_loss, total_disc_loss, real_disc_loss, fake_disc_loss

