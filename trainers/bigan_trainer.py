from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from time import sleep
from time import time
from utils.evaluations import do_prc


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
                loc=0.0,
                scale=1.0,
                size=[self.config.data_loader.test_batch, self.noise_dim],
            )
            image_eval = self.sess.run(image)
            feed_dict = {
                self.model.image_tensor: image_eval,
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
        self.logger.warn("Testing evaluation...")
        scores = []
        inference_time = []
        true_labels = []
        # Create the scores
        test_loop = tqdm(range(self.config.data_loader.num_iter_per_test))
        for _ in test_loop:
            test_batch_begin = time()
            test_batch, test_labels = self.sess.run(
                [self.data.test_image, self.data.test_label]
            )
            test_loop.refresh()  # to show immediately the update
            sleep(0.01)
            noise = np.random.normal(
                loc=0.0,
                scale=1.0,
                size=[self.config.data_loader.test_batch, self.noise_dim],
            )
            feed_dict = {
                self.model.image_tensor: test_batch,
                self.model.noise_tensor: noise,
                self.model.is_training: False,
            }
            scores += self.sess.run(
                self.model.list_scores, feed_dict=feed_dict
            ).tolist()
            inference_time.append(time() - test_batch_begin)
            true_labels += test_labels.tolist()
        true_labels = np.asarray(true_labels)
        inference_time = np.mean(inference_time)
        self.logger.info("Testing: Mean inference time is {:4f}".format(inference_time))
        prc_auc = do_prc(
            scores,
            true_labels,
            file_name=r"bigan/mnist/{}/{}/{}".format(
                self.config.trainer.loss_method,
                self.config.trainer.weight,
                self.config.trainer.label,
            ),
            directory=r"results/bigan/mnist/{}/{}/".format(
                self.config.trainer.loss_method, self.config.trainer.weight
            ),
        )
        self.logger.info("Testing | PRC AUC = {:.4f}".format(prc_auc))

    def train_step(self, image, cur_epoch):
        noise = np.random.normal(
            loc=0.0, scale=1.0, size=[self.batch_size, self.noise_dim]
        )
        # Train the discriminator
        image_eval = self.sess.run(image)
        feed_dict = {
            self.model.image_tensor: image_eval,
            self.model.noise_tensor: noise,
            self.model.is_training: True,
        }
        # Train Discriminator
        _, ld, sm_d = self.sess.run(
            [
                self.model.train_dis_op,
                self.model.loss_discriminator,
                self.model.sum_op_dis,
            ],
            feed_dict=feed_dict,
        )

        # Train Generator and Encoder
        noise = np.random.normal(
            loc=0.0, scale=1.0, size=[self.batch_size, self.noise_dim]
        )
        feed_dict = {
            self.model.image_tensor: image_eval,
            self.model.noise_tensor: noise,
            self.model.is_training: True,
        }
        _, _, le, lg, sm_g = self.sess.run(
            [
                self.model.train_gen_op,
                self.model.trian_enc_op,
                self.model.loss_encoder,
                self.model.loss_generator,
                self.model.sum_op_gen,
            ],
            feed_dict=feed_dict,
        )

        return lg, ld, le, sm_g, sm_d
