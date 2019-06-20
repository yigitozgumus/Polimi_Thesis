from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from time import sleep
from time import time
from utils.evaluations import save_results


class ALAD_Trainer(BaseTrain):
    def __init__(self, sess, model, data, config, summarizer):
        super(ALAD_Trainer, self).__init__(sess, model, data, config, summarizer)
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
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        # Attach the epoch loop to a variable
        begin = time()
        # Make the loop of the epoch iterations
        loop = tqdm(range(self.config.data_loader.num_iter_per_epoch))
        # Define Losses
        gen_losses = []
        enc_losses = []
        disc_losses = []
        disc_xz_losses = []
        disc_xx_losses = []
        disc_zz_losses = []
        summaries = []
        # Get the current epoch counter
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        image = self.data.image
        for _ in loop:
            loop.set_description("Epoch:{}".format(cur_epoch + 1))
            loop.refresh()  # to show immediately the update
            sleep(0.01)
            # Compute the main losses
            lg, le, ld, ldxz, ldxx, ldzz, summary = self.train_step(image, cur_epoch)
            gen_losses.append(lg)
            enc_losses.append(le)
            disc_losses.append(ld)
            disc_xz_losses.append(ldxz)
            disc_xx_losses.append(ldxx)
            disc_zz_losses.append(ldzz)
            summaries.append(summary)
        self.logger.info("Epoch {} terminated".format(cur_epoch))
        self.summarizer.add_tensorboard(step=cur_epoch, summaries=summaries)
        # Check for reconstruction
        if cur_epoch % self.config.log.frequency_test == 0:
            noise = np.random.normal(
                loc=0.0, scale=1.0, size=[self.config.data_loader.batch_size, self.noise_dim]
            )
            real_noise, fake_noise = self.generate_noise(
                self.config.trainer.include_noise, cur_epoch
            )
            image_eval = self.sess.run(image)
            feed_dict = {
                self.model.image_tensor: image_eval,
                self.model.noise_tensor: noise,
                self.model.real_noise: real_noise,
                self.model.fake_noise: fake_noise,
                self.model.is_training: False,
            }
            reconstruction = self.sess.run(self.model.sum_op_im, feed_dict=feed_dict)
            self.summarizer.add_tensorboard(step=cur_epoch, summaries=[reconstruction])
        # Get the means of the loss values to display
        gl_m = np.mean(gen_losses)
        el_m = np.mean(enc_losses)
        dl_m = np.mean(disc_losses)
        dlxz_m = np.mean(disc_xz_losses)
        dlxx_m = np.mean(disc_xx_losses)
        dlzz_m = np.mean(disc_zz_losses)
        if self.config.trainer.allow_zz:
            self.logger.info(
                "Epoch {} | time = {} | loss gen = {:4f} | loss enc = {:4f} | "
                "loss dis = {:4f} | loss dis xz = {:4f} | loss dis xx = {:4f} | "
                "loss dis zz = {:4f}".format(
                    cur_epoch, time() - begin, gl_m, el_m, dl_m, dlxz_m, dlxx_m, dlzz_m
                )
            )
        else:
            self.logger.info(
                "Epoch {} | time = {} | loss gen = {:4f} | loss enc = {:4f} | "
                "loss dis = {:4f} | loss dis xz = {:4f} | loss dis xx = {:4f} | ".format(
                    cur_epoch, time() - begin, gl_m, el_m, dl_m, dlxz_m, dlxx_m
                )
            )
        self.model.save(self.sess)
        # Early Stopping
        if (
            cur_epoch + 1
        ) % self.config.trainer.frequency_eval == 0 and self.config.trainer.enable_early_stop:
            valid_loss = 0
            image_valid = self.sess.run(self.data.valid_image)
            noise = np.random.normal(
                loc=0.0, scale=1.0, size=[image_valid.shape[0], self.noise_dim]
            )
            feed_dict = {
                self.model.image_tensor: image_valid,
                self.model.noise_tensor: noise,
                self.model.is_training: False,
            }
            vl, lat = self.sess.run(
                [self.model.rec_error_valid, self.model.rec_z], feed_dict=feed_dict
            )
            valid_loss += vl
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
        noise = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, self.noise_dim])
        true_labels, generated_labels = self.generate_labels(
            self.config.trainer.soft_labels, self.config.trainer.flip_labels
        )
        # Train the discriminator
        real_noise, fake_noise = self.generate_noise(self.config.trainer.include_noise, cur_epoch)
        image_eval = self.sess.run(image)
        feed_dict = {
            self.model.image_tensor: image_eval,
            self.model.noise_tensor: noise,
            self.model.generated_labels: generated_labels,
            self.model.true_labels: true_labels,
            self.model.real_noise: real_noise,
            self.model.fake_noise: fake_noise,
            self.model.is_training: True,
        }
        _, _, _, ld, ldxz, ldxx, ldzz = self.sess.run(
            [
                self.model.train_dis_op_xz,
                self.model.train_dis_op_xx,
                self.model.train_dis_op_zz,
                self.model.loss_discriminator,
                self.model.dis_loss_xz,
                self.model.dis_loss_xx,
                self.model.dis_loss_zz,
            ],
            feed_dict=feed_dict,
        )
        # Train the Generator and Encoder
        noise = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, self.noise_dim])
        true_labels, generated_labels = self.generate_labels(
            self.config.trainer.soft_labels, self.config.trainer.flip_labels
        )
        real_noise, fake_noise = self.generate_noise(self.config.trainer.include_noise, cur_epoch)
        feed_dict = {
            self.model.image_tensor: image_eval,
            self.model.noise_tensor: noise,
            self.model.generated_labels: generated_labels,
            self.model.true_labels: true_labels,
            self.model.real_noise: real_noise,
            self.model.fake_noise: fake_noise,
            self.model.is_training: True,
        }
        _, _, le, lg = self.sess.run(
            [
                self.model.train_gen_op,
                self.model.train_enc_op,
                self.model.loss_encoder,
                self.model.loss_generator,
            ],
            feed_dict=feed_dict,
        )

        if self.config.log.enable_summary:
            sm = self.sess.run(self.model.sum_op, feed_dict=feed_dict)
        else:
            sm = None

        return lg, le, ld, ldxz, ldxx, ldzz, sm

    def test_epoch(self):
        # Evaluation for the testing
        self.logger.info("Testing evaluation...")
        scores_ch = []
        scores_l1 = []
        scores_l2 = []
        scores_fm = []
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
                self.model.image_tensor: test_batch,
                self.model.noise_tensor: noise,
                self.model.is_training: False,
            }
            scores_ch += self.sess.run(self.model.score_ch, feed_dict=feed_dict).tolist()
            scores_l1 += self.sess.run(self.model.score_l1, feed_dict=feed_dict).tolist()
            scores_l2 += self.sess.run(self.model.score_l2, feed_dict=feed_dict).tolist()
            scores_fm += self.sess.run(self.model.score_fm, feed_dict=feed_dict).tolist()
            summaries += self.sess.run(self.model.sum_op_im_test, feed_dict=feed_dict)
            inference_time.append(time() - test_batch_begin)
            true_labels += test_labels.tolist()
        scores_ch = np.asarray(scores_ch)
        scores_l1 = np.asarray(scores_l1)
        scores_l2 = np.asarray(scores_l2)
        scores_fm = np.asarray(scores_fm)
        true_labels = np.asarray(true_labels)
        inference_time = np.mean(inference_time)
        self.summarizer.add_tensorboard(step=cur_epoch, summaries=summaries, summarizer="test")
        self.logger.info("Testing: Mean inference time is {:4f}".format(inference_time))
        # TODO BATCH FILL ?
        model = "alad_sn{}_dzz{}".format(
            self.config.trainer.do_spectral_norm, self.config.trainer.allow_zz
        )
        random_seed = 42
        label = 0
        step = self.sess.run(self.model.global_step_tensor)
        percentiles = np.asarray(self.config.trainer.percentiles)
        save_results(
            self.config.log.result_dir,
            scores_ch,
            true_labels,
            model,
            self.config.data_loader.dataset_name,
            "ch",
            "dzzenabled{}".format(self.config.trainer.allow_zz),
            label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )
        save_results(
            self.config.log.result_dir,
            scores_l1,
            true_labels,
            model,
            self.config.data_loader.dataset_name,
            "l1",
            "dzzenabled{}".format(self.config.trainer.allow_zz),
            label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )
        save_results(
            self.config.log.result_dir,
            scores_l2,
            true_labels,
            model,
            self.config.data_loader.dataset_name,
            "l2",
            "dzzenabled{}".format(self.config.trainer.allow_zz),
            label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )
        save_results(
            self.config.log.result_dir,
            scores_fm,
            true_labels,
            model,
            self.config.data_loader.dataset_name,
            "fm",
            "dzzenabled{}".format(self.config.trainer.allow_zz),
            label,
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
