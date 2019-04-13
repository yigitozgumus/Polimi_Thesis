from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from time import sleep
import time


class GANTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, summarizer):
        super(GANTrainer, self).__init__(sess, model, data, config, summarizer)

    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        # Attach the epoch loop to a variable
        loop = tqdm(range(self.config.data_loader.num_iter_per_epoch))
        # Define the lists for summaries and losses
        gen_losses = []
        disc_losses = []
        summaries = []

        # Get the current epoch counter
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        self.sess.run(self.data.iterator.initializer)
        image = self.data.image
        for _ in loop:
            loop.set_description("Epoch:{}".format(cur_epoch + 1))
            loop.refresh()  # to show immediately the update
            sleep(0.01)
            gen_loss, disc_loss, summary = self.train_step(image, cur_epoch=cur_epoch)
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)
            summaries.append(summary)
        # write the summaries
        self.summarizer.add_tensorboard(cur_epoch, summaries=summary)
        # Compute the means of the losses
        gen_loss_m = np.mean(gen_losses)
        disc_loss_m = np.mean(disc_losses)
        # Generate images between epochs to evaluate
        if cur_epoch % self.config.log.frequency_test == 0:
            noise = np.random.normal(
                loc=0.0,
                scale=1.0,
                size=[
                    self.config.data_loader.test_batch,
                    self.config.trainer.noise_dim,
                ],
            )
            image_eval = self.sess.run(image)
            feed_dict = {
                self.model.image_input: image_eval,
                self.model.noise_tensor: noise,
                self.model.is_training: False,
            }
            reconstruction = self.sess.run(
                self.model.summary_image, feed_dict=feed_dict
            )
            self.summarizer.add_tensorboard(step=cur_epoch, summaries=[reconstruction])

        if cur_epoch % self.config.log.show_steps == 0 or cur_epoch == 1:
            self.logger.info(
                "Epoch {}, Generator Loss: {}, Discriminator Loss: {}".format(
                    cur_epoch + 1, gen_loss_m, disc_loss_m
                )
            )

        self.model.save(self.sess)

    def train_step(self, image, cur_epoch):
        # Generate noise from uniform  distribution between -1 and 1
        # New Noise Generation
        # noise = np.random.uniform(-1., 1.,size=[self.config.batch_size, self.config.noise_dim])
        sigma = max(0.75 * (10.0 - cur_epoch) / (10), 0.05)
        noise = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=[self.config.data_loader.batch_size, self.config.trainer.noise_dim],
        )
        true_labels, generated_labels = self.generate_labels(
            self.config.trainer.soft_labels
        )
        # Instance noise additions

        if self.config.trainer.include_noise:
            # If we want to add this is will add the noises
            real_noise = np.random.normal(
                scale=sigma,
                size=[self.config.data_loader.batch_size]
                + self.config.trainer.image_dims,
            )
            fake_noise = np.random.normal(
                scale=sigma,
                size=[self.config.data_loader.batch_size]
                + self.config.trainer.image_dims,
            )
        else:
            # Otherwise we are just going to add zeros which will not break anything
            real_noise = np.zeros(
                ([self.config.data_loader.batch_size] + self.config.trainer.image_dims)
            )
            fake_noise = np.zeros(
                ([self.config.data_loader.batch_size] + self.config.trainer.image_dims)
            )
        # Evaluation of the image
        image_eval = self.sess.run(image)
        # Construct the Feed Dictionary
        # Train the Discriminator on both real and fake images
        feed_dict = {
            self.model.noise_tensor: noise,
            self.model.image_input: image_eval,
            self.model.true_labels: true_labels,
            self.model.generated_labels: generated_labels,
            self.model.real_noise: real_noise,
            self.model.fake_noise: fake_noise,
            self.model.is_training: True,
        }
        _, disc_loss = self.sess.run(
            [self.model.train_disc, self.model.total_disc_loss], feed_dict=feed_dict
        )
        # Train the Generator and get the summaries
        # Re create the noise for the generator
        noise = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=[self.config.data_loader.batch_size, self.config.trainer.noise_dim],
        )
        if self.config.include_noise:
            # If we want to add this is will add the noises
            fake_noise = np.random.normal(
                scale=sigma,
                size=[self.config.data_loader.batch_size]
                + self.config.trainer.image_dims,
            )
        else:
            # Otherwise we are just going to add zeros which will not break anything
            fake_noise = np.zeros(
                ([self.config.data_loader.batch_size] + self.config.trainer.image_dims)
            )
        true_labels, generated_labels = self.generate_labels(
            self.config.trainer.soft_labels
        )
        feed_dict = {
            self.model.noise_tensor: noise,
            self.model.image_input: image_eval,
            self.model.true_labels: true_labels,
            self.model.generated_labels: generated_labels,
            self.model.fake_noise: fake_noise,
            self.model.is_training: True,
        }
        _, gen_loss = self.sess.run(
            [self.model.train_gen, self.model.gen_loss], feed_dict=feed_dict
        )
        if self.config.log.enable_summary:
            sm = self.sess.run(self.model.summary_all, feed_dict=feed_dict)
        else:
            sm = None

        return gen_loss, disc_loss, sm

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
