from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from time import sleep
from time import time
from utils.evaluations import do_roc, save_results


class BIGANTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, summarizer):
        super(BIGANTrainer, self).__init__(sess, model, data, config, summarizer)
        self.batch_size = self.config.data_loader.batch_size
        self.noise_dim = self.config.trainer.noise_dim
        self.img_dims = self.config.trainer.image_dims
        # Inititalize the train Dataset Iterator
        self.sess.run(self.data.iterator.initializer)
        # Initialize the test Dataset Iterator
        self.sess.run(self.data.test_iterator.initializer)
        if self.config.data_loader.validation:
            self.sess.run(self.data.valid_iterator.initializer)
            self.best_valid_loss = 0
            self.nb_without_improvements = 0

    def train_epoch(self):
        begin = time()
        # Attach the epoch loop to a variable
        loop = tqdm(range(self.config.data_loader.num_iter_per_epoch))
        # Define the lists for summaries and losses
        gen_losses = []
        disc_losses = []
        enc_losses = []
        summaries = []

        # Get the current epoch counter
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        image = self.data.image
        for _ in loop:
            loop.set_description("Epoch:{}".format(cur_epoch + 1))
            loop.refresh()  # to show immediately the update
            sleep(0.01)
            gen, dis, enc, sum_g, sum_d = self.train_step(image, cur_epoch)
            gen_losses.append(gen)
            disc_losses.append(dis)
            enc_losses.append(enc)
            summaries.append(sum_g)
            summaries.append(sum_d)
        self.logger.info("Epoch {} terminated".format(cur_epoch))
        self.summarizer.add_tensorboard(step=cur_epoch, summaries=summaries)
        # Check for reconstruction
        if cur_epoch % self.config.log.frequency_test == 0:
            noise = np.random.normal(
                loc=0.0, scale=1.0, size=[self.config.data_loader.test_batch, self.noise_dim]
            )
            image_eval = self.sess.run(image)
            feed_dict = {
                self.model.image_input: image_eval,
                self.model.noise_tensor: noise,
                self.model.is_training: False,
            }
            reconstruction = self.sess.run(self.model.sum_op_im, feed_dict=feed_dict)
            self.summarizer.add_tensorboard(step=cur_epoch, summaries=[reconstruction])
        # Get the means of the loss values to display
        gen_m = np.mean(gen_losses)
        dis_m = np.mean(disc_losses)
        enc_m = np.mean(enc_losses)
        self.logger.info(
            "Epoch: {} | time = {} s | loss gen= {:4f} | loss dis = {:4f} | loss enc = {:4f}".format(
                cur_epoch, time() - begin, gen_m, dis_m, enc_m
            )
        )
        # Save the model state
        self.model.save(self.sess)
        if (
            cur_epoch + 1
        ) % self.config.trainer.frequency_eval == 0 and self.config.trainer.enable_early_stop:
            valid_loss = 0
            image_valid = self.sess.run(self.data.valid_image)
            noise = np.random.normal(
                loc=0.0, scale=1.0, size=[self.config.data_loader.test_batch, self.noise_dim]
            )
            feed_dict = {
                self.model.noise_tensor: noise,
                self.model.image_input: image_valid,
                self.model.is_training: False,
            }
            vl = self.sess.run([self.model.rec_error_valid], feed_dict=feed_dict)
            valid_loss += vl[0]
            if self.config.log.enable_summary:
                sm = self.sess.run(self.model.sum_op_valid, feed_dict=feed_dict)
                self.summarizer.add_tensorboard(step=cur_epoch, summaries=[sm], summarizer="valid")

            self.logger.info("Validation: valid loss {:.4f}".format(valid_loss))
            if (
                valid_loss < self.best_valid_loss
                or cur_epoch == self.config.trainer.frequency_eval - 1
            ):
                self.best_valid_loss = valid_loss
                self.logger.info(
                    "Best model - valid loss = {:.4f} - saving...".format(self.best_valid_loss)
                )
                # Save the model state
                self.model.save(self.sess)
                self.nb_without_improvements = 0
            else:
                self.nb_without_improvements += self.config.trainer.frequency_eval
            if self.nb_without_improvements > self.config.trainer.patience:
                self.patience_lost = True
                self.logger.warning(
                    "Early stopping at epoch {} with weights from epoch {}".format(
                        cur_epoch, cur_epoch - self.nb_without_improvements
                    )
                )

    def test_epoch(self):
        self.logger.warn("Testing evaluation...")
        scores_1 = []
        scores_2 = []
        inference_time = []
        true_labels = []
        summaries = []
        # Create the scores
        test_loop = tqdm(range(self.config.data_loader.num_iter_per_test))
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        for _ in test_loop:
            test_batch_begin = time()
            test_batch, test_labels = self.sess.run([self.data.test_image, self.data.test_label])
            test_loop.refresh()  # to show immediately the update
            sleep(0.01)
            noise = np.random.normal(
                loc=0.0, scale=1.0, size=[self.config.data_loader.test_batch, self.noise_dim]
            )
            feed_dict = {
                self.model.image_input: test_batch,
                self.model.noise_tensor: noise,
                self.model.is_training: False,
            }
            scores_1 += self.sess.run(self.model.list_scores_1, feed_dict=feed_dict).tolist()
            scores_2 += self.sess.run(self.model.list_scores_2, feed_dict=feed_dict).tolist()
            summaries += self.sess.run([self.model.sum_op_im_test], feed_dict=feed_dict)
            inference_time.append(time() - test_batch_begin)
            true_labels += test_labels.tolist()
        # Since the higher anomaly score indicates the anomalous one, and we inverted the labels to show that
        # normal images are 0 meaning that contains no anomaly and anomalous images are 1 meaning that it contains
        # an anomalous region, we first scale the scores and then invert them to match the scores
        scores_1 = np.asarray(scores_1)
        scores_2 = np.asarray(scores_2)
        true_labels = np.asarray(true_labels)
        inference_time = np.mean(inference_time)
        self.summarizer.add_tensorboard(step=cur_epoch, summaries=summaries, summarizer="test")
        self.logger.info("Testing: Mean inference time is {:4f}".format(inference_time))
        step = self.sess.run(self.model.global_step_tensor)
        percentiles = np.asarray(self.config.trainer.percentiles)
        save_results(
            self.config.log.result_dir,
            scores_1,
            true_labels,
            self.config.model.name,
            self.config.data_loader.dataset_name,
            "fm_1",
            "paper",
            self.config.trainer.label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )
        save_results(
            self.config.log.result_dir,
            scores_2,
            true_labels,
            self.config.model.name,
            self.config.data_loader.dataset_name,
            "fm_2",
            "paper",
            self.config.trainer.label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )

    def train_step(self, image, cur_epoch):
        image_eval = self.sess.run(image)
        # Train the discriminator
        ld, sm_d = 0, None
        if self.config.trainer.mode == "standard":
            disc_iters = 1
        else:
            disc_iters = self.config.trainer.critic_iters
        for _ in range(disc_iters):
            noise = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, self.noise_dim])
            true_labels, generated_labels = self.generate_labels(
                self.config.trainer.soft_labels, self.config.trainer.flip_labels
            )
            real_noise, fake_noise = self.generate_noise(
                self.config.trainer.include_noise, cur_epoch
            )
            feed_dict = {
                self.model.image_input: image_eval,
                self.model.noise_tensor: noise,
                self.model.generated_labels: generated_labels,
                self.model.true_labels: true_labels,
                self.model.real_noise: real_noise,
                self.model.fake_noise: fake_noise,
                self.model.is_training: True,
            }
            # Train Discriminator
            _, ld, sm_d = self.sess.run(
                [self.model.train_dis_op, self.model.loss_discriminator, self.model.sum_op_dis],
                feed_dict=feed_dict,
            )
            if self.config.trainer.mode == "wgan":
                _ = self.sess.run(self.model.clip_disc_weights)
        # Train Generator and Encoder
        noise = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, self.noise_dim])
        true_labels, generated_labels = self.generate_labels(
            self.config.trainer.soft_labels, self.config.trainer.flip_labels
        )
        real_noise, fake_noise = self.generate_noise(self.config.trainer.include_noise, cur_epoch)
        feed_dict = {
            self.model.image_input: image_eval,
            self.model.noise_tensor: noise,
            self.model.generated_labels: generated_labels,
            self.model.true_labels: true_labels,
            self.model.real_noise: real_noise,
            self.model.fake_noise: fake_noise,
            self.model.is_training: True,
        }
        _, _, le, lg, sm_g = self.sess.run(
            [
                self.model.train_gen_op,
                self.model.train_enc_op,
                self.model.loss_encoder,
                self.model.loss_generator,
                self.model.sum_op_gen,
            ],
            feed_dict=feed_dict,
        )

        return lg, np.mean(ld), le, sm_g, sm_d

    def generate_labels(self, soft_labels, flip_labels):

        if not soft_labels:
            true_labels = np.ones((self.config.data_loader.batch_size, 1))
            generated_labels = np.zeros((self.config.data_loader.batch_size, 1))
        else:
            generated_labels = np.zeros(
                (self.config.data_loader.batch_size, 1)
            ) + np.random.uniform(low=0.0, high=0.1, size=[self.config.data_loader.batch_size, 1])
            flipped_idx = np.random.choice(
                np.arange(len(generated_labels)),
                size=int(self.config.trainer.noise_probability * len(generated_labels)),
            )
            generated_labels[flipped_idx] = 1 - generated_labels[flipped_idx]
            true_labels = np.ones((self.config.data_loader.batch_size, 1)) - np.random.uniform(
                low=0.0, high=0.1, size=[self.config.data_loader.batch_size, 1]
            )
            flipped_idx = np.random.choice(
                np.arange(len(true_labels)),
                size=int(self.config.trainer.noise_probability * len(true_labels)),
            )
            true_labels[flipped_idx] = 1 - true_labels[flipped_idx]
        if flip_labels:
            return generated_labels, true_labels
        else:
            return true_labels, generated_labels

    def generate_noise(self, include_noise, cur_epoch):
        sigma = max(0.75 * (10.0 - cur_epoch) / (10), 0.05)
        if include_noise:
            # If we want to add this is will add the noises
            real_noise = np.random.normal(
                scale=sigma,
                size=[self.config.data_loader.batch_size] + self.config.trainer.image_dims,
            )
            fake_noise = np.random.normal(
                scale=sigma,
                size=[self.config.data_loader.batch_size] + self.config.trainer.image_dims,
            )
        else:
            # Otherwise we are just going to add zeros which will not break anything
            real_noise = np.zeros(
                ([self.config.data_loader.batch_size] + self.config.trainer.image_dims)
            )
            fake_noise = np.zeros(
                ([self.config.data_loader.batch_size] + self.config.trainer.image_dims)
            )
        return real_noise, fake_noise
