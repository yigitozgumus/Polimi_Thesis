from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from time import sleep
from time import time
from utils.evaluations import save_results


class EBGANTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, summarizer):
        super(EBGANTrainer, self).__init__(sess, model, data, config, summarizer)
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
        self.logger.info(
            "Epoch: {} | time = {} s | loss gen= {:4f} | loss dis = {:4f} ".format(
                cur_epoch, time() - begin, gen_m, dis_m
            )
        )
        # Save the model state
        self.model.save(self.sess)
        # TODO Validation

    def test_epoch(self):
        self.logger.warn("Testing evaluation...")
        scores_im1 = []
        scores_im2 = []
        scores_z1 = []
        scores_z2 = []
        inference_time = []
        true_labels = []
        # Create the scores
        test_loop = tqdm(range(self.config.data_loader.num_iter_per_test))
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
            scores_im1 += self.sess.run(self.model.img_score_l1, feed_dict=feed_dict).tolist()
            scores_im2 += self.sess.run(self.model.img_score_l2, feed_dict=feed_dict).tolist()
            scores_z1 += self.sess.run(self.model.z_score_l1, feed_dict=feed_dict).tolist()
            scores_z2 += self.sess.run(self.model.z_score_l2, feed_dict=feed_dict).tolist()
            inference_time.append(time() - test_batch_begin)
            true_labels += test_labels.tolist()
        scores_im1 = np.asarray(scores_im1)
        scores_im2 = np.asarray(scores_im2)
        scores_z1 = np.asarray(scores_z1)
        scores_z2 = np.asarray(scores_z2)

        true_labels = np.asarray(true_labels)
        inference_time = np.mean(inference_time)
        self.logger.info("Testing: Mean inference time is {:4f}".format(inference_time))
        step = self.sess.run(self.model.global_step_tensor)
        percentiles = np.asarray(self.config.trainer.percentiles)
        save_results(
            self.config.log.result_dir,
            scores_im1,
            true_labels,
            self.config.model.name,
            self.config.data_loader.dataset_name,
            "im1",
            "paper",
            self.config.trainer.label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )
        save_results(
            self.config.log.result_dir,
            scores_im2,
            true_labels,
            self.config.model.name,
            self.config.data_loader.dataset_name,
            "im2",
            "paper",
            self.config.trainer.label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )
        save_results(
            self.config.log.result_dir,
            scores_z1,
            true_labels,
            self.config.model.name,
            self.config.data_loader.dataset_name,
            "z1",
            "paper",
            self.config.trainer.label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )
        save_results(
            self.config.log.result_dir,
            scores_z2,
            true_labels,
            self.config.model.name,
            self.config.data_loader.dataset_name,
            "z2",
            "paper",
            self.config.trainer.label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )

    def train_step(self, image, cur_epoch):
        # Train Generator
        if self.config.trainer.mode == "standard":
            gen_iters = 1
        else:
            gen_iters = 3
        lg_t = 0
        sm_g = 0
        ld_t = 0
        for _ in range(gen_iters):
            image_eval = self.sess.run(image)
            noise = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, self.noise_dim])
            feed_dict = {
                self.model.image_input: image_eval,
                self.model.noise_tensor: noise,
                self.model.is_training: True,
            }
            _, lg, sm_g = self.sess.run(
                [self.model.train_gen_op, self.model.loss_generator, self.model.sum_op_gen],
                feed_dict=feed_dict,
            )
            lg_t += lg
        image_eval = self.sess.run(image)
        if self.config.trainer.mode == "standard":
            disc_iters = 1
        else:
            disc_iters = self.config.trainer.critic_iters
        for _ in range(disc_iters):
            noise = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, self.noise_dim])
            feed_dict = {
                self.model.image_input: image_eval,
                self.model.noise_tensor: noise,
                self.model.is_training: True,
            }
            _, ld, sm_d = self.sess.run(
                [self.model.train_dis_op, self.model.loss_discriminator, self.model.sum_op_dis],
                feed_dict=feed_dict,
            )
            ld_t += ld

        return np.mean(lg_t), np.mean(ld_t), sm_g, sm_d
