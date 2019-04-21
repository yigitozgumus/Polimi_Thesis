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
            feed_dict = {self.model.image_input: image_eval, self.model.is_training: False}
            reconstruction = self.sess.run(self.model.sum_op_im, feed_dict=feed_dict)
            self.summarizer.add_tensorboard(step=cur_epoch, summaries=[reconstruction])
        # Get the means of the loss values to display
        gen_m = np.mean(gen_losses)
        dis_m = np.mean(disc_losses)
        self.logger.info(
            "Epoch: {} | time = {} s | loss gen= {:4f} | loss dis = {:4f} }".format(
                cur_epoch, time() - begin, gen_m, dis_m
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
            test_batch, test_labels = self.sess.run([self.data.test_image, self.data.test_label])
            test_loop.refresh()  # to show immediately the update
            sleep(0.01)
            feed_dict = {self.model.image_input: test_batch, self.model.is_training: False}
            scores += self.sess.run(self.model.score, feed_dict=feed_dict).tolist()
            inference_time.append(time() - test_batch_begin)
            true_labels += test_labels.tolist()
        true_labels = np.asarray(true_labels)
        self.logger.info("Testing: Mean inference time is {:4f}".format(inference_time))
        scores = np.asarray(scores)
        scores_scaled = (scores - min(scores)) / (max(scores) - min(scores))
        random_seed = 42  # TODO
        step = self.sess.run(self.model.global_step_tensor)
        save_results(
            self.config.log.result_dir,
            scores_scaled,
            true_labels,
            self.config.model.name,
            self.config.data_loader.dataset_name,
            "fm",
            "paper",
            self.config.trainer.label,
            random_seed,
            step,
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
        feed_dict = {
            self.model.image_tensor: image_eval,
            self.model.generated_labels: generated_labels,
            self.model.true_labels: true_labels,
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
        feed_dict = {
            self.model.image_tensor: image_eval,
            self.model.generated_labels: generated_labels,
            self.model.true_labels: true_labels,
            self.model.is_training: True,
        }
        _, lg, sm_g = self.sess.run(
            [self.model.train_gen_op, self.model.gen_loss_total, self.model.sum_op_gen],
            feed_dict=feed_dict,
        )

        return lg, ld, sm_g, sm_d

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
