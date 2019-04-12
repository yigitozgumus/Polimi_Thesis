from base.base_train_eager import BaseTrainEager
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from time import sleep


class GANTrainerEager(BaseTrainEager):
    def __init__(self, model, data, config, summarizer):
        super(GANTrainerEager, self).__init__(model, data, config, summarizer)
        self.summarizer.train_summary_writer.set_as_default()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        loop = self.data.dataset
        # Define the lists for summaries and losses
        gen_losses = []
        total_disc_losses = []

        # Get the current epoch
        cur_epoch = self.model.cur_epoch_tensor.numpy()
        for image_batch in loop:
            gen_loss, total_dl = self.train_step(image_batch, cur_epoch=cur_epoch)
            gen_losses.append(gen_loss)
            total_disc_losses.append(total_dl)
        # write the summaries
        # self.logger.summarize(cur_epoch, summaries=summaries)
        # Compute the means of the losses
        gen_loss_m = np.mean(gen_losses)
        disc_loss_t = np.mean(total_disc_losses)
        with self.summarizer.train_summary_writer.as_default():
            tf.summary.scalar("Generator_Loss", gen_loss_m)
            tf.summary.scalar("Total_Discriminator_Loss", disc_loss_t)
        # Generate images between epochs to evaluate
        rand_noise = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=[
                self.config.log.num_example_imgs_to_generate,
                self.config.trainer.noise_dim,
            ],
        )
        progress_images = self.model.generator(rand_noise, training=False)
        self.save_generated_images(progress_images, cur_epoch)

        if cur_epoch % self.config.log.show_steps == 0 or cur_epoch == 1:
            self.logger.info(
                "Epoch {} --\nGenerator Loss: {}\nDiscriminator Loss: {}\n".format(
                    cur_epoch, gen_loss_m, disc_loss_t
                )
            )
        self.model.save()

    @tf.function
    def train_step(self, image, cur_epoch):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        # Generate noise from uniform  distribution between -1 and 1
        sigma = max(0.75 * (10.0 - cur_epoch) / (10), 0.05)
        noise = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=[self.config.data_loader.batch_size, self.config.trainer.noise_dim],
        )
        # Soft Label Generation
        true_labels, generated_labels = self.generate_labels(
            self.config.trainer.soft_labels
        )

        # Train the Discriminator
        gradients, disc_loss = self.disc_compute_gradients(
            self.model.generator,
            self.model.discriminator,
            image,
            noise,
            true_labels,
            generated_labels,
        )
        self.apply_gradients(
            self.model.discriminator_optimizer,
            gradients,
            self.model.discriminator.trainable_variables,
        )

        # Train the Generator
        noise = tf.random.normal(
            mean=0.0,
            stddev=1.0,
            shape=[self.config.data_loader.batch_size, self.config.trainer.noise_dim],
        )
        gradients, gen_loss = self.gen_compute_gradients(
            self.model.generator, self.model.discriminator, noise
        )
        self.apply_gradients(
            self.model.generator_optimizer,
            gradients,
            self.model.generator.trainable_variables,
        )

        return gen_loss, disc_loss

    def gen_compute_loss(self, model_gen, model_disc, noise):
        generated_sample = model_gen(noise, training=True)
        stacked_gan = model_disc(generated_sample, training=True)
        loss = self.cross_entropy(tf.zeros_like(stacked_gan), stacked_gan)
        return loss

    def disc_compute_loss(
        self, model_gen, model_disc, image, noise, true_labels, generated_labels
    ):
        generated_sample = model_gen(noise, training=True)
        disc_real = model_disc(image, training=True)
        disc_fake = model_disc(generated_sample, training=True)

        disc_loss_real = self.cross_entropy(true_labels, disc_real)

        disc_loss_fake = self.cross_entropy(generated_labels, disc_fake)

        total_disc_loss = disc_loss_real + disc_loss_fake

        return total_disc_loss

    def gen_compute_gradients(self, model_gen, model_disc, noise):
        with tf.GradientTape() as tape:
            loss = self.gen_compute_loss(model_gen, model_disc, noise)
        return tape.gradient(loss, model_gen.trainable_variables), loss

    def disc_compute_gradients(
        self, model_gen, model_disc, image, noise, true_labels, generated_labels
    ):
        with tf.GradientTape() as tape:
            loss = self.disc_compute_loss(
                model_gen, model_disc, image, noise, true_labels, generated_labels
            )
        return tape.gradient(loss, model_disc.trainable_variables), loss

    def apply_gradients(self, optimizer, gradients, variables):
        optimizer.apply_gradients(zip(gradients, variables))

    def generate_labels(self, soft_labels):

        if not soft_labels:
            true_labels = np.ones((self.config.data_loader.batch_size, 1))
            generated_labels = np.zeros((self.config.data_loader.batch_size, 1))
        else:
            true_labels = np.zeros(
                (self.config.data_loader.batch_size, 1)
            ) + np.random.uniform(
                low=0.0, high=0.1, size=[self.config.data_loader.batch_size, 1]
            )
            flipped_idx = np.random.choice(
                np.arange(len(true_labels)),
                size=int(self.config.trainer.noise_probability * len(true_labels)),
            )
            true_labels[flipped_idx] = 1 - true_labels[flipped_idx]
            generated_labels = np.ones(
                (self.config.data_loader.batch_size, 1)
            ) - np.random.uniform(
                low=0.0, high=0.1, size=[self.config.data_loader.batch_size, 1]
            )
            flipped_idx = np.random.choice(
                np.arange(len(generated_labels)),
                size=int(self.config.trainer.noise_probability * len(generated_labels)),
            )
            generated_labels[flipped_idx] = 1 - generated_labels[flipped_idx]

        return true_labels, generated_labels
