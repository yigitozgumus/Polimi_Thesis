from base.base_train_sequential import BaseTrainSequential
from tqdm import tqdm
import numpy as np
from time import sleep
from time import time
from utils.evaluations import save_results


class SENCEBGANTrainer_Denoiser(BaseTrainSequential):
    def __init__(self, sess, model, data, config, logger):
        super(SENCEBGANTrainer_Denoiser, self).__init__(sess, model, data, config, logger)
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

    def train_epoch_gan(self):
        # Attach the epoch loop to a variable
        begin = time()
        # Make the loop of the epoch iterations
        loop = tqdm(range(self.config.data_loader.num_iter_per_epoch))
        gen_losses = []
        disc_losses = []
        summaries = []
        image = self.data.image
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        for _ in loop:
            loop.set_description("Epoch:{}".format(cur_epoch + 1))
            loop.refresh()  # to show immediately the update
            sleep(0.01)
            lg, ld, sum_g, sum_d = self.train_step_gan(image, cur_epoch)
            gen_losses.append(lg)
            disc_losses.append(ld)
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
                self.model.is_training_gen: False,
            }
            reconstruction = self.sess.run(self.model.sum_op_im_1, feed_dict=feed_dict)
            self.summarizer.add_tensorboard(step=cur_epoch, summaries=[reconstruction])
        gen_m = np.mean(gen_losses)
        dis_m = np.mean(disc_losses)
        self.logger.info(
            "Epoch: {} | time = {} s | loss gen= {:4f} | loss dis = {:4f} ".format(
                cur_epoch, time() - begin, gen_m, dis_m
            )
        )
        self.model.save(self.sess)

    def train_epoch_enc_gen(self):
        # Attach the epoch loop to a variable
        begin = time()
        # Make the loop of the epoch iterations
        loop = tqdm(range(self.config.data_loader.num_iter_per_epoch))
        enc_losses = []
        disc_xx_losses = []
        summaries = []
        image = self.data.image
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        for _ in loop:
            loop.set_description("Epoch:{}".format(cur_epoch + 1))
            loop.refresh()  # to show immediately the update
            sleep(0.01)
            le, sum_e, ldxx = self.train_step_enc_gen(image, cur_epoch)
            enc_losses.append(le)
            if self.config.trainer.enable_disc_xx:
                disc_xx_losses.append(ldxx)
            summaries.append(sum_e)
        self.logger.info("Epoch {} terminated".format(cur_epoch))
        self.summarizer.add_tensorboard(step=cur_epoch, summaries=summaries, summarizer="valid")
        # Check for reconstruction
        if cur_epoch % self.config.log.frequency_test == 0:
            noise = np.random.normal(
                loc=0.0, scale=1.0, size=[self.config.data_loader.test_batch, self.noise_dim]
            )
            image_eval = self.sess.run(image)
            feed_dict = {
                self.model.image_input: image_eval,
                self.model.noise_tensor: noise,
                self.model.is_training_gen: False,
                self.model.is_training_enc_g: False,
                self.model.is_training_enc_r: False,
                self.model.is_training_dis: False,
            }
            reconstruction = self.sess.run(self.model.sum_op_im_2, feed_dict=feed_dict)
            self.summarizer.add_tensorboard(
                step=cur_epoch, summaries=[reconstruction], summarizer="valid"
            )
        enc_m = np.mean(enc_losses)
        if self.config.trainer.enable_disc_xx:
            dis_xx_m = np.mean(disc_xx_losses)
            self.logger.info(
                "Epoch: {} | time = {} s | loss enc generation= {:4f}  | loss dis xx = {:4f}".format(
                    cur_epoch, time() - begin, enc_m, dis_xx_m
                )
            )
        else:
            self.logger.info(
                "Epoch: {} | time = {} s | loss enc generation= {:4f}  ".format(
                    cur_epoch, time() - begin, enc_m
                )
            )
        self.model.save(self.sess)

    def train_epoch_enc_rec(self):
        # Attach the epoch loop to a variable
        begin = time()
        # Make the loop of the epoch iterations
        loop = tqdm(range(self.config.data_loader.num_iter_per_epoch))
        enc_losses = []
        summaries = []
        image = self.data.image
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        for _ in loop:
            loop.set_description("Epoch:{}".format(cur_epoch + 1))
            loop.refresh()  # to show immediately the update
            sleep(0.01)
            le, sum_e = self.train_step_enc_rec(image, cur_epoch)
            enc_losses.append(le)
            summaries.append(sum_e)
        self.logger.info("Epoch {} terminated".format(cur_epoch))
        self.summarizer.add_tensorboard(step=cur_epoch, summaries=summaries, summarizer="valid_2")
        enc_m = np.mean(enc_losses)
        self.logger.info(
            "Epoch: {} | time = {} s | loss Denoiser= {:4f}  ".format(
                cur_epoch, time() - begin, enc_m
            )
        )
        self.model.save(self.sess)

    def train_step_gan(self, image, cur_epoch):
        ld_t, lg_t, sm_g, sm_d = [], [], None, None
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
                self.model.is_training_gen: True,
                self.model.is_training_dis: True,
                self.model.is_training_enc_g: False,
                self.model.is_training_enc_r: False,
            }
            _, ld, sm_d = self.sess.run(
                [self.model.train_dis_op, self.model.loss_discriminator, self.model.sum_op_dis],
                feed_dict=feed_dict,
            )
            ld_t.append(ld)

        if self.config.trainer.mode == "standard":
            gen_iters = 1
        else:
            gen_iters = 3
        for _ in range(gen_iters):
            image_eval = self.sess.run(image)
            noise = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, self.noise_dim])
            feed_dict = {
                self.model.image_input: image_eval,
                self.model.noise_tensor: noise,
                self.model.is_training_gen: True,
                self.model.is_training_dis: True,
                self.model.is_training_enc_g: False,
                self.model.is_training_enc_r: False,
            }
            _, lg, sm_g = self.sess.run(
                [self.model.train_gen_op, self.model.loss_generator, self.model.sum_op_gen],
                feed_dict=feed_dict,
            )
            lg_t.append(lg)

        return np.mean(lg_t), np.mean(ld_t), sm_g, sm_d

    def train_step_enc_gen(self, image, cur_epoch):
        image_eval = self.sess.run(image)
        noise = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, self.noise_dim])
        ldxx = 0
        feed_dict = {
            self.model.image_input: image_eval,
            self.model.noise_tensor: noise,
            self.model.is_training_gen: False,
            self.model.is_training_dis: False,
            self.model.is_training_enc_g: True,
            self.model.is_training_enc_r: False,
        }
        if self.config.trainer.enable_disc_xx:
            _, le, sm_e, = self.sess.run(
                [self.model.train_enc_g_op, self.model.loss_encoder_g, self.model.sum_op_enc_g],
                feed_dict=feed_dict,
            )
            _, ldxx = self.sess.run(
                [self.model.train_dis_op_xx, self.model.dis_loss_xx], feed_dict=feed_dict
            )
            # Additional generator discriminator training
           # _ = self.sess.run([self.model.train_gen_op], feed_dict=feed_dict)
           # _ = self.sess.run([self.model.train_dis_op], feed_dict=feed_dict)
        else:
            _, le, sm_e = self.sess.run(
                [self.model.train_enc_g_op, self.model.loss_encoder_g, self.model.sum_op_enc_g],
                feed_dict=feed_dict,
            )
        return le, sm_e, ldxx

    def train_step_enc_rec(self, image, cur_epoch):
        image_eval = self.sess.run(image)
        noise = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, self.noise_dim])
        noise_2 = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=[self.config.data_loader.batch_size] + self.config.trainer.image_dims,
        )
       
        feed_dict = {
            self.model.image_input: image_eval,
            self.model.denoiser_noise: noise_2,
            self.model.is_training_gen: False,
            self.model.is_training_dis: False,
            self.model.is_training_enc_g: False,
            self.model.is_training_enc_r: True,
        }
        _, le, sm_e = self.sess.run(
            [self.model.train_den_op, self.model.den_loss, self.model.sum_op_den],
            feed_dict=feed_dict,
        )
        return le, sm_e

    def test_epoch(self):
        self.logger.warn("Testing evaluation...")
        scores_im1 = []
        scores_im2 = []
        scores_comb = []
        scores_mask1 = []
        scores_mask2 = []
        scores_pipe = []
        scores_pipe_2 = []
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
                self.model.is_training_gen: False,
                self.model.is_training_dis: False,
                self.model.is_training_enc_g: False,
                self.model.is_training_enc_r: False,
            }
            scores_im1 += self.sess.run(self.model.img_score_l1, feed_dict=feed_dict).tolist()
            scores_im2 += self.sess.run(self.model.img_score_l2, feed_dict=feed_dict).tolist()
            scores_comb += self.sess.run(self.model.score_comb, feed_dict=feed_dict).tolist()
            scores_mask1 += self.sess.run(self.model.mask_score_1, feed_dict=feed_dict).tolist()
            scores_mask2 += self.sess.run(self.model.mask_score_2, feed_dict=feed_dict).tolist()
            scores_pipe += self.sess.run(self.model.pipe_score, feed_dict=feed_dict).tolist()
            scores_pipe_2 += self.sess.run(self.model.pipe_score_2, feed_dict=feed_dict).tolist()
            if self.config.trainer.enable_disc_xx:
                # scores_final_3 += self.sess.run(
                #     self.model.final_score_3, feed_dict=feed_dict
                # ).tolist()
                scores_final_4 += self.sess.run(
                    self.model.final_score_4, feed_dict=feed_dict
                ).tolist()
            if self.config.trainer.enable_disc_zz:
                # scores_final_5 += self.sess.run(
                #     self.model.final_score_5, feed_dict=feed_dict
                # ).tolist()
                scores_final_6 += self.sess.run(
                    self.model.final_score_6, feed_dict=feed_dict
                ).tolist()
            inference_time.append(time() - test_batch_begin)
            true_labels += test_labels.tolist()
        scores_im1 = np.asarray(scores_im1)
        scores_im2 = np.asarray(scores_im2)
        scores_comb = np.asarray(scores_comb)
        scores_pipe = np.asarray(scores_pipe)
        scores_pipe_2 = np.asarray(scores_pipe_2)
        scores_mask1 = np.asarray(scores_mask1)
        scores_mask2 = np.asarray(scores_mask2)
        if self.config.trainer.enable_disc_xx:
            #scores_final_3 = np.asarray(scores_final_3)
            scores_final_4 = np.asarray(scores_final_4)
        if self.config.trainer.enable_disc_zz:
            #scores_final_5 = np.asarray(scores_final_5)
            scores_final_6 = np.asarray(scores_final_6)
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
            scores_comb,
            true_labels,
            self.config.model.name,
            self.config.data_loader.dataset_name,
            "comb",
            "paper",
            self.config.trainer.label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )
        save_results(
            self.config.log.result_dir,
            scores_mask1,
            true_labels,
            self.config.model.name,
            self.config.data_loader.dataset_name,
            "mask_1",
            "paper",
            self.config.trainer.label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )
        save_results(
            self.config.log.result_dir,
            scores_mask2,
            true_labels,
            self.config.model.name,
            self.config.data_loader.dataset_name,
            "mask_2",
            "paper",
            self.config.trainer.label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )
        save_results(
            self.config.log.result_dir,
            scores_pipe,
            true_labels,
            self.config.model.name,
            self.config.data_loader.dataset_name,
            "scores_pipe_1",
            "paper",
            self.config.trainer.label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )
        save_results(
            self.config.log.result_dir,
            scores_pipe_2,
            true_labels,
            self.config.model.name,
            self.config.data_loader.dataset_name,
            "scores_pipe_2",
            "paper",
            self.config.trainer.label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )
