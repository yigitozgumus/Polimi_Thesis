from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from time import sleep
from time import time
from utils.evaluations import save_results


class ANOGAN_Trainer(BaseTrain):
    def __init__(self, sess, model, data, config, summarizer):
        super(ANOGAN_Trainer, self).__init__(sess, model, data, config, summarizer)
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
        # Attach the epoch loop to a variable
        begin = time()
        # Make the loop of the epoch iterations
        loop = tqdm(range(self.config.data_loader.num_iter_per_epoch))
        # Define Losses
        gen_losses = []
        disc_losses = []
        summaries = []
        image = self.data.image
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        for _ in loop:
            loop.set_description("Epoch:{}".format(cur_epoch + 1))
            loop.refresh()  # to show immediately the update
            sleep(0.01)
            ld, lg, sm = self.train_step(image, cur_epoch)
            gen_losses.append(lg)
            disc_losses.append(ld)
            summaries.append(sm)
        self.logger.info("Epoch {} terminated".format(cur_epoch))
        gl_m = np.mean(gen_losses)
        dl_m = np.mean(disc_losses)
        self.summarizer.add_tensorboard(step=cur_epoch, summaries=summaries)
        # Check for reconstruction
        if cur_epoch % self.config.log.frequency_test == 0:
            image_eval = self.sess.run(image)
            noise = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, self.noise_dim])
            feed_dict = {
                self.model.image_input: image_eval,
                self.model.noise_tensor: noise,
                self.model.is_training: False,
            }
            reconstruction = self.sess.run(self.model.sum_op_im, feed_dict=feed_dict)
            self.summarizer.add_tensorboard(step=cur_epoch, summaries=[reconstruction])
        self.logger.info("Epoch terminated")
        self.logger.info(
            "Epoch %d | time = %ds | loss gen = %.4f | loss dis = %.4f "
            % (cur_epoch, time() - begin, gl_m, dl_m)
        )
        # Save the model state
        # self.model.save(self.sess)
        if (
            cur_epoch + 1
        ) % self.config.trainer.frequency_eval == 0 and self.config.trainer.enable_early_stop:
            valid_loss = 0
            image_valid = self.sess.run(self.data.valid_image)
            noise = np.random.normal(
                loc=0.0, scale=1.0, size=[image_valid.shape[0], self.noise_dim]
            )
            feed_dict = {
                self.model.image_input: image_valid,
                self.model.noise_tensor: noise,
                self.model.is_training: False,
            }
            vl = self.sess.run([self.model.rec_error_valid], feed_dict=feed_dict)
            valid_loss += vl
            if self.config.log.enable_summary:
                sm_im = self.sess.run(self.model.sum_op_im, feed_dict=feed_dict)
                sm_vl = self.sess.run(self.model.sum_op_valid, feed_dict=feed_dict)
                self.summarizer.add_tensorboard(
                    step=cur_epoch, summaries=[sm_im, sm_vl], summarizer="valid"
                )

            self.logger.info("Validation: valid loss {:.4f}".format(valid_loss))
            if (
                self.config.trainer.difference <= self.best_valid_loss - valid_loss
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
        # Evaluation for the testing
        self.logger.info("Testing evaluation...")
        rect_x, rec_error, latent, scores = [], [], [], []
        inference_time = []
        true_labels = []
        # Create the scores
        test_loop = tqdm(range(self.config.data_loader.num_iter_pre_test))
        for _ in test_loop:
            begin_val_batch = time()
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
            for _ in range(self.config.trainer.steps_number):
                _ = self.sess.run(self.model.invert_op, feed_dict=feed_dict)

            brect_x, brec_error, bscores, blatent = self.sess.run(
                [
                    self.model.rec_gen_ema,
                    self.model.reconstruction_score,
                    self.model.loss_invert,
                    self.model.z_optim,
                ],
                feed_dict=feed_dict,
            )
            rect_x.append(brect_x)
            rec_error.append(brec_error)
            scores.append(bscores)
            latent.append(blatent)
            self.sess.run(self.model.reinit_test_graph_op)

            inference_time.append(time() - begin_val_batch)
            true_labels += test_labels.tolist()
        true_labels = np.asarray(true_labels)
        inference_time = np.mean(inference_time)
        self.logger.info("Testing: Mean inference time is {:4f}".format(inference_time))
        rect_x = np.concatenate(rect_x, axis=0)
        rec_error = np.concatenate(rec_error, axis=0)
        scores = np.concatenate(scores, axis=0)
        latent = np.concatenate(latent, axis=0)
        step = self.sess.run(self.model.global_step_tensor)
        save_results(
            self.config.log.result_dir,
            scores,
            true_labels,
            "anogan",
            self.config.data_loader.dataset_name,
            "cross_e",
            self.config.trainer.weight,
            self.config.trainer.label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
        )

    def train_step(self, image, cur_epoch):
        noise = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, self.noise_dim])
        true_labels, generated_labels = self.generate_labels(
            self.config.trainer.soft_labels, self.config.trainer.flip_labels
        )
        # Train the discriminator
        image_eval = self.sess.run(image)
        feed_dict = {
            self.model.image_input: image_eval,
            self.model.noise_tensor: noise,
            self.model.true_labels: true_labels,
            self.model.generated_labels: generated_labels,
            self.model.is_training: True,
        }
        _, ld = self.sess.run(
            [self.model.train_dis_op, self.model.total_disc_loss], feed_dict=feed_dict
        )

        noise = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, self.noise_dim])
        true_labels, generated_labels = self.generate_labels(
            self.config.trainer.soft_labels, self.config.trainer.flip_labels
        )
        feed_dict = {
            self.model.image_input: image_eval,
            self.model.noise_tensor: noise,
            self.model.true_labels: true_labels,
            self.model.generated_labels: generated_labels,
            self.model.is_training: True,
        }
        # Train the generator
        _, lg = self.sess.run([self.model.train_gen_op, self.model.gen_loss], feed_dict=feed_dict)

        if self.config.log.enable_summary:
            sm = self.sess.run(self.model.sum_op, feed_dict=feed_dict)
        else:
            sm = None

        return ld, lg, sm

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
