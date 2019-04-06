from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from time import sleep
from time import time


class ALAD_Trainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(ALAD_Trainer, self).__init__(sess, model, data, config, logger)
        self.batch_size = self.config.data_loader.batch_size
        self.noise_dim = self.config.trainer.noise_dim
        self.img_dims = self.config.trainer.image_dims

    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        # Attach the epoch loop to a variable
        begin = time()
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
        for _ in loop:
            loop.set_description("Epoch:{}".format(cur_epoch + 1))
            loop.refresh()  # to show immediately the update
            sleep(0.01)
            lg, le, ld, ldxz, ldxx, ldzz, summary = self.train_step(self.data.image, cur_epoch)
            gen_losses.append(lg)
            enc_losses.append(le)
            disc_losses.append(ld)
            disc_xz_losses.append(ldxz)
            disc_xx_losses.append(ldxx)
            disc_zz_losses.append(ldzz)
        self.logger.info("Epoch {} terminated".format(cur_epoch))
        gl_m = np.mean(gen_losses)
        el_m = np.mean(enc_losses)
        dl_m = np.mean(disc_losses)
        dlxz_m = np.mean(disc_xz_losses)
        dlxx_m = np.mean(disc_xx_losses)
        dlzz_m = np.mean(disc_zz_losses)
        if self.config.trainer.allow_zz:
            print("Epoch {} | time = {} | loss gen = {:4f} | loss enc = {:4f} | "
                  "loss dis = {:4f} | loss dis xz = {:4f} | loss dis xx = {:4f} | "
                  "loss dis zz = {:4f}".format(cur_epoch, time() - begin, gl_m,
                                               el_m, dl_m, dlxz_m, dlxx_m, dlzz_m))
        else:
            print("Epoch {} | time = {} | loss gen = {:4f} | loss enc = {:4f} | "
                  "loss dis = {:4f} | loss dis xz = {:4f} | loss dis xx = {:4f} | "
                  .format(cur_epoch, time() - begin, gl_m,
                          el_m, dl_m, dlxz_m, dlxx_m))

        self.model.save(self.sess)
        self.logger.warn("Testing evaluation...")
        #TODO

    def train_step(self, image, cur_epoch):
        """
       implement the logic of the train step
       - run the tensorflow session
       - return any metrics you need to summarize
       """
        noise = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, self.noise_dim])
        # Train the discriminator
        feed_dict = {self.model.image_tensor: image,
                     self.model.noise_tensor: noise,
                     self.model.is_training: True}
        self.logger.debug("Session for the Discriminator")
        _, _, _, ld, ldxz, ldxx, ldzz = self.sess.run([self.model.train_dis_op_xz,
                                                       self.model.train_dis_op_xx,
                                                       self.model.train_dis_op_zz,
                                                       self.model.loss_discriminator,
                                                       self.model.dis_loss_xz,
                                                       self.model.dis_loss_xx,
                                                       self.model.dis_loss_zz],
                                                      feed_dict=feed_dict)
        # Train the Generator and Encoder
        noise = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, self.noise_dim])
        feed_dict = {self.model.image_tensor: image,
                     self.model.noise_tensor: noise,
                     self.model.is_training: True}
        _, _, le, lg = self.sess.run([self.model.train_gen_op,
                                      self.model.train_enc_op,
                                      self.model.loss_encoder,
                                      self.model.loss_generator],
                                     feed_dict=feed_dict)

        if self.config.log.enable_summary:
            sm = self.sess.run(self.model.sum_op, feed_dict=feed_dict)
        else:
            sm = None
            # TODO add reconstruction code

        return lg, le, ld, ldxz, ldxx, ldzz, sm
