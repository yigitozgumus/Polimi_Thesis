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


        # Get the current epoch
        cur_epoch = self.model.cur_epoch_tensor.numpy()
        for image in self.data.dataset:
            gen_loss, total_dl = self.train_step(image.numpy().astype(np.float32), cur_epoch=cur_epoch)
            gen_losses.append(gen_loss)
            total_disc_losses.append(total_dl)

        # write the summaries
        #self.logger.summarize(cur_epoch, summaries=summaries)
        # Compute the means of the losses
        gen_loss_m = np.mean(gen_losses)
        disc_loss_t = np.mean(total_disc_losses)
        with tf.contrib.summary.record_summaries_every_n_global_steps(1):
            tf.contrib.summary.scalar("Generator_Loss", gen_loss_m)
            tf.contrib.summary.scalar("Total_Discriminator_Loss", disc_loss_t)

        # Generate images between epochs to evaluate
        rand_noise = np.random.normal(
            loc=0.0, scale=1.0,
            size=[self.config.num_example_imgs_to_generate, self.config.noise_dim]).astype(np.float32)
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
        noise = np.random.normal(loc=0.0, scale=1.0, size=[self.config.batch_size, self.config.noise_dim]).astype(np.float32)
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

        # Train the Discriminator
        gradients, disc_loss = self.disc_compute_gradients(
            self.model.generator,
            self.model.discriminator,
            image,
            noise,
            true_labels,
            generated_labels)
        self.apply_gradients(self.model.discriminator_optimizer,
                             gradients,
                             self.model.discriminator.trainable_variables)
        

        # Train the Generator
        noise = np.random.normal(loc=0.0, scale=1.0, size=[self.config.batch_size, self.config.noise_dim]).astype(np.float32)
        gradients, gen_loss = self.gen_compute_gradients(
            self.model.generator,
            self.model.discriminator,
            noise)
        self.apply_gradients(self.model.generator_optimizer,
                             gradients,
                             self.model.generator.trainable_variables)

        return gen_loss, disc_loss

    def gen_compute_loss(self,model_gen,model_disc, noise):
        generated_sample = model_gen(noise, training=True)
        stacked_gan = model_disc(generated_sample, training=True)
        loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(stacked_gan), stacked_gan)
        return loss

    def disc_compute_loss(self,model_gen,model_disc, image, noise,true_labels, generated_labels):
        generated_sample = model_gen(noise, training=True)
        disc_real = model_disc(image, training=True)
        disc_fake = model_disc(generated_sample, training=True)

        disc_loss_real = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=true_labels, logits=disc_real)

        disc_loss_fake = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=generated_labels,logits=disc_fake)

        total_disc_loss = disc_loss_real + disc_loss_fake

        return total_disc_loss

    def gen_compute_gradients(self, model_gen, model_disc, noise):
        with tf.GradientTape() as tape:
            loss = self.gen_compute_loss(model_gen,model_disc,noise)
        return tape.gradient(loss, model_gen.trainable_variables), loss

    def disc_compute_gradients(self,model_gen, model_disc, image, noise,true_labels, generated_labels):
        with tf.GradientTape() as tape:
            loss = self.disc_compute_loss(model_gen,model_disc, image, noise,true_labels, generated_labels)
        return tape.gradient(loss, model_disc.trainable_variables), loss

    def apply_gradients(self,optimizer, gradients, variables):
        optimizer.apply_gradients(zip(gradients, variables))


