from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from time import sleep
from time import time
from utils.evaluations import save_results


class GANomalyTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, summarizer):
        super(GANomalyTrainer, self).__init__(sess, model, data, config, summarizer)
        # This values are added as variable becaouse they are used a lot and changing it become difficult over time.
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
        summaries = []
        # Get the current epoch counter
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        image = self.data.image
        for _ in loop:
            loop.set_description("Epoch:{}".format(cur_epoch + 1))
            loop.refresh()  # to show immediately the update
            sleep(0.01)
            gen, dis, sum_g, sum_d = self.train_step(image, cur_epoch)
            gen_losses.append(gen)
            disc_losses.append(dis)
            summaries.append(sum_g)
            summaries.append(sum_d)
        self.logger.info("Epoch {} terminated".format(cur_epoch))
        self.summarizer.add_tensorboard(step=cur_epoch, summaries=summaries)
        # Check for reconstruction
        if cur_epoch % self.config.log.frequency_test == 0:
            image_eval = self.sess.run(image)
            real_noise, fake_noise = self.generate_noise(False, cur_epoch)
            feed_dict = {
                self.model.image_input: image_eval,
                self.model.real_noise: real_noise,
                self.model.fake_noise: fake_noise,
                self.model.is_training: False,
            }
            reconstruction = self.sess.run(self.model.sum_op_im, feed_dict=feed_dict)
            self.summarizer.add_tensorboard(step=cur_epoch, summaries=[reconstruction])
        # Get the means of the loss values to display
        gen_m = np.mean(gen_losses)
        dis_m = np.mean(disc_losses)
        self.logger.info(
            "Epoch: {} | time = {} s | loss gen= {:4f} | loss dis = {:4f}".format(
                cur_epoch, time() - begin, gen_m, dis_m
            )
        )
        # Save the model state
        self.model.save(self.sess)

        if (
            cur_epoch + 1
        ) % self.config.trainer.frequency_eval == 0 and self.config.trainer.enable_early_stop:
            valid_loss = 0
            image_valid = self.sess.run(self.data.valid_image)

            feed_dict = {self.model.image_input: image_valid, self.model.is_training: False}
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

    def train_step(self, image, cur_epoch):
        """
          implement the logic of the train step
          - run the tensorflow session
          - return any metrics you need to summarize
        """
        true_labels, generated_labels = self.generate_labels(
            self.config.trainer.soft_labels, self.config.trainer.flip_labels
        )
        # Train the discriminator
        image_eval = self.sess.run(image)
        real_noise, fake_noise = self.generate_noise(self.config.trainer.include_noise, cur_epoch)
        feed_dict = {
            self.model.image_input: image_eval,
            self.model.generated_labels: generated_labels,
            self.model.true_labels: true_labels,
            self.model.real_noise: real_noise,
            self.model.fake_noise: fake_noise,
            self.model.is_training: True,
        }
        _, ld, sm_d = self.sess.run(
            [self.model.train_dis_op, self.model.loss_discriminator, self.model.sum_op_dis],
            feed_dict=feed_dict,
        )

        # Train Generator
        true_labels, generated_labels = self.generate_labels(
            self.config.trainer.soft_labels, self.config.trainer.flip_labels
        )
        real_noise, fake_noise = self.generate_noise(self.config.trainer.include_noise, cur_epoch)
        feed_dict = {
            self.model.image_input: image_eval,
            self.model.generated_labels: generated_labels,
            self.model.true_labels: true_labels,
            self.model.real_noise: real_noise,
            self.model.fake_noise: fake_noise,
            self.model.is_training: True,
        }
        _, lg, sm_g = self.sess.run(
            [self.model.train_gen_op, self.model.gen_loss_total, self.model.sum_op_gen],
            feed_dict=feed_dict,
        )

        return lg, ld, sm_g, sm_d

    def test_epoch(self):
        self.logger.warn("Testing evaluation...")
        scores = []
        inference_time = []
        true_labels = []
        # Create the scores
        test_loop = tqdm(range(self.config.data_loader.num_iter_per_test))
        for _ in test_loop:
            test_batch_begin = time()
            test_batch, test_labels = self.sess.run([self.data.test_image, self.data.test_label])
            test_loop.refresh()  # to show immediately the update
            sleep(0.01)
            feed_dict = {self.model.image_input: test_batch, self.model.is_training: False}
            scores += self.sess.run(self.model.score, feed_dict=feed_dict).tolist()
            inference_time.append(time() - test_batch_begin)
            true_labels += test_labels.tolist()
        true_labels = np.asarray(true_labels)
        inference_time = np.mean(inference_time)
        self.logger.info("Testing: Mean inference time is {:4f}".format(inference_time))
        scores = np.asarray(scores)
        step = self.sess.run(self.model.global_step_tensor)
        percentiles = np.asarray(self.config.trainer.percentiles)
        save_results(
            self.config.log.result_dir,
            scores,
            true_labels,
            self.config.model.name,
            self.config.data_loader.dataset_name,
            "fm",
            "paper",
            self.config.trainer.label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )

    def generate_labels(self, soft_labels, flip_labels):

        if not soft_labels:
            true_labels = np.ones((self.config.data_loader.batch_size, 1))
            generated_labels = np.zeros((self.config.data_loader.batch_size, 1))
        else:
            generated_labels = np.zeros(
                (self.config.data_loader.batch_size, 1)
            ) + np.random.uniform(low=0.0, high=0.1, size=[self.config.data_loader.batch_size, 1])
            # flipped_idx = np.random.choice(
            #     np.arange(len(generated_labels)),
            #     size=int(self.config.trainer.noise_probability * len(generated_labels)),
            # )
            # generated_labels[flipped_idx] = 1 - generated_labels[flipped_idx]
            true_labels = np.ones((self.config.data_loader.batch_size, 1)) - np.random.uniform(
                low=0.0, high=0.1, size=[self.config.data_loader.batch_size, 1]
            )
            # flipped_idx = np.random.choice(
            #     np.arange(len(true_labels)),
            #     size=int(self.config.trainer.noise_probability * len(true_labels)),
            # )
            # true_labels[flipped_idx] = 1 - true_labels[flipped_idx]
        if flip_labels:
            return generated_labels, true_labels
        else:
            return true_labels, generated_labels

    def generate_noise(self, include_noise, cur_epoch, mode=1):
        sigma = max(1.25 * (10.0 - cur_epoch) / (10), 1)
        real_noise = np.zeros(
            ([self.config.data_loader.batch_size] + self.config.trainer.image_dims)
        )
        if include_noise:
            # If we want to add this is will add the noises
            fake_noise = np.random.normal(
                scale=sigma,
                size=[self.config.data_loader.batch_size] + self.config.trainer.image_dims,
            )
            if mode == 2:
                real_noise = np.random.normal(
                    scale=sigma,
                    size=[self.config.data_loader.batch_size] + self.config.trainer.image_dims,
                )

        else:
            # Otherwise we are just going to add zeros which will not break anything
            fake_noise = np.zeros(
                ([self.config.data_loader.batch_size] + self.config.trainer.image_dims)
            )
        return real_noise, fake_noise
