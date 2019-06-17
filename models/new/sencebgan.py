import tensorflow as tf

from base.base_model import BaseModel
from utils.alad_utils import get_getter
import utils.alad_utils as sn


class SENCEBGAN(BaseModel):
    def __init__(self, config):
        super(SENCEBGAN, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        ############################################################################################
        # INIT
        ############################################################################################
        # Kernel initialization for the convolutions
        if self.config.trainer.init_type == "normal":
            self.init_kernel = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        elif self.config.trainer.init_type == "xavier":
            self.init_kernel = tf.contrib.layers.xavier_initializer(
                uniform=False, seed=None, dtype=tf.float32
            )
        # Placeholders
        self.is_training_gen = tf.placeholder(tf.bool)
        self.is_training_dis = tf.placeholder(tf.bool)
        self.is_training_enc_g = tf.placeholder(tf.bool)
        self.is_training_enc_r = tf.placeholder(tf.bool)
        self.image_input = tf.placeholder(
            tf.float32, shape=[None] + self.config.trainer.image_dims, name="x"
        )
        self.noise_tensor = tf.placeholder(
            tf.float32, shape=[None, self.config.trainer.noise_dim], name="noise"
        )
        ############################################################################################
        # MODEL
        ############################################################################################
        self.logger.info("Building training graph...")
        with tf.variable_scope("SENCEBGAN"):
            # First training part
            # G(z) ==> x'
            with tf.variable_scope("Generator_Model"):
                self.image_gen = self.generator(self.noise_tensor)
            # Discriminator outputs
            with tf.variable_scope("Discriminator_Model"):
                self.embedding_real, self.decoded_real = self.discriminator(
                    self.image_input, do_spectral_norm=self.config.trainer.do_spectral_norm
                )
                self.embedding_fake, self.decoded_fake = self.discriminator(
                    self.image_gen, do_spectral_norm=self.config.trainer.do_spectral_norm
                )
            # Second training part
            # E(x) ==> z'
            with tf.variable_scope("Encoder_G_Model"):
                self.image_encoded = self.encoder_g(self.image_input)
            # G(z') ==> G(E(x)) ==> x''
            with tf.variable_scope("Generator_Model"):
                self.image_gen_enc = self.generator(self.image_encoded)
            # Discriminator outputs
            with tf.variable_scope("Discriminator_Model"):
                self.embedding_enc_fake, self.decoded_enc_fake = self.discriminator(
                    self.image_gen_enc, do_spectral_norm=self.config.trainer.do_spectral_norm
                )
                self.embedding_enc_real, self.decoded_enc_real = self.discriminator(
                    self.image_input, do_spectral_norm=self.config.trainer.do_spectral_norm
                )
            with tf.variable_scope("Discriminator_Model_XX"):
                self.im_logit_real, self.im_f_real = self.discriminator_xx(
                    self.image_input,
                    self.image_input,
                    do_spectral_norm=self.config.trainer.do_spectral_norm,
                )
                self.im_logit_fake, self.im_f_fake = self.discriminator_xx(
                    self.image_input,
                    self.image_gen_enc,
                    do_spectral_norm=self.config.trainer.do_spectral_norm,
                )
            # Third training part
            with tf.variable_scope("Encoder_G_Model"):
                self.image_encoded_r = self.encoder_g(self.image_input)

            with tf.variable_scope("Generator_Model"):
                self.image_gen_enc_r = self.generator(self.image_encoded_r)

            with tf.variable_scope("Encoder_R_Model"):
                self.image_ege = self.encoder_r(self.image_gen_enc_r)

            with tf.variable_scope("Discriminator_Model_ZZ"):
                self.z_logit_real, self.z_f_real = self.discriminator_zz(
                    self.image_encoded_r,
                    self.image_encoded_r,
                    do_spectral_norm=self.config.trainer.do_spectral_norm,
                )
                self.z_logit_fake, self.z_f_fake = self.discriminator_zz(
                    self.image_encoded_r,
                    self.image_ege,
                    do_spectral_norm=self.config.trainer.do_spectral_norm,
                )

        ############################################################################################
        # LOSS FUNCTIONS
        ############################################################################################
        with tf.name_scope("Loss_Functions"):
            with tf.name_scope("Generator_Discriminator"):
                # Discriminator Loss
                if self.config.trainer.mse_mode == "norm":
                    self.disc_loss_real = tf.reduce_mean(
                        self.mse_loss(
                            self.decoded_real,
                            self.image_input,
                            mode="norm",
                            order=self.config.trainer.order,
                        )
                    )
                    self.disc_loss_fake = tf.reduce_mean(
                        self.mse_loss(
                            self.decoded_fake,
                            self.image_gen,
                            mode="norm",
                            order=self.config.trainer.order,
                        )
                    )
                elif self.config.trainer.mse_mode == "mse":
                    self.disc_loss_real = self.mse_loss(
                        self.decoded_real,
                        self.image_input,
                        mode="mse",
                        order=self.config.trainer.order,
                    )
                    self.disc_loss_fake = self.mse_loss(
                        self.decoded_fake,
                        self.image_gen,
                        mode="mse",
                        order=self.config.trainer.order,
                    )
                self.loss_discriminator = (
                    tf.math.maximum(self.config.trainer.disc_margin - self.disc_loss_fake, 0)
                    + self.disc_loss_real
                )
                # Generator Loss
                pt_loss = 0
                if self.config.trainer.pullaway:
                    pt_loss = self.pullaway_loss(self.embedding_fake)
                self.loss_generator = self.disc_loss_fake + self.config.trainer.pt_weight * pt_loss
                # New addition to enforce visual similarity
                delta_noise = self.embedding_real - self.embedding_fake
                delta_flat = tf.layers.Flatten()(delta_noise)
                loss_noise_gen = tf.reduce_mean(
                    tf.norm(delta_flat, ord=2, axis=1, keepdims=False)
                    )
                self.loss_generator += (0.1 * loss_noise_gen)

            with tf.name_scope("Encoder_G"):
                if self.config.trainer.mse_mode == "norm":
                    self.loss_enc_rec = tf.reduce_mean(
                        self.mse_loss(
                            self.image_gen_enc,
                            self.image_input,
                            mode="norm",
                            order=self.config.trainer.order,
                        )
                    )
                    self.loss_enc_f = tf.reduce_mean(
                        self.mse_loss(
                            self.decoded_enc_real,
                            self.decoded_enc_fake,
                            mode="norm",
                            order=self.config.trainer.order,
                        )
                    )
                elif self.config.trainer.mse_mode == "mse":
                    self.loss_enc_rec = tf.reduce_mean(
                        self.mse_loss(
                            self.image_gen_enc,
                            self.image_input,
                            mode="mse",
                            order=self.config.trainer.order,
                        )
                    )
                    self.loss_enc_f = tf.reduce_mean(
                        self.mse_loss(
                            self.embedding_enc_real,
                            self.embedding_enc_fake,
                            mode="mse",
                            order=self.config.trainer.order,
                        )
                    )
                self.loss_encoder_g = (
                    self.loss_enc_rec + self.config.trainer.encoder_f_factor * self.loss_enc_f
                )
                if self.config.trainer.enable_disc_xx:
                    self.enc_xx_real = tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.im_logit_real, labels=tf.zeros_like(self.im_logit_real)
                    )
                    self.enc_xx_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.im_logit_fake, labels=tf.ones_like(self.im_logit_fake)
                    )
                    self.enc_loss_xx = tf.reduce_mean(self.enc_xx_real + self.enc_xx_fake)
                    self.loss_encoder_g += self.enc_loss_xx

            with tf.name_scope("Encoder_R"):
                if self.config.trainer.mse_mode == "norm":
                    self.loss_encoder_r = tf.reduce_mean(
                        self.mse_loss(
                            self.image_ege,
                            self.image_encoded_r,
                            mode="norm",
                            order=self.config.trainer.order,
                        )
                    )

                elif self.config.trainer.mse_mode == "mse":
                    self.loss_encoder_r = tf.reduce_mean(
                        self.mse_loss(
                            self.image_ege,
                            self.image_encoded_r,
                            mode="mse",
                            order=self.config.trainer.order,
                        )
                    )

                if self.config.trainer.enable_disc_zz:
                    self.enc_zz_real = tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.z_logit_real, labels=tf.zeros_like(self.z_logit_real)
                    )
                    self.enc_zz_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.z_logit_fake, labels=tf.ones_like(self.z_logit_fake)
                    )
                    self.enc_loss_zz = tf.reduce_mean(self.enc_zz_real + self.enc_zz_fake)
                    self.loss_encoder_r += self.enc_loss_zz

            if self.config.trainer.enable_disc_xx:
                with tf.name_scope("Discriminator_XX"):
                    self.loss_xx_real = tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.im_logit_real, labels=tf.ones_like(self.im_logit_real)
                    )
                    self.loss_xx_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.im_logit_fake, labels=tf.zeros_like(self.im_logit_fake)
                    )
                    self.dis_loss_xx = tf.reduce_mean(self.loss_xx_real + self.loss_xx_fake)
            if self.config.trainer.enable_disc_zz:
                with tf.name_scope("Discriminator_ZZ"):
                    self.loss_zz_real = tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.z_logit_real, labels=tf.ones_like(self.z_logit_real)
                    )
                    self.loss_zz_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.z_logit_fake, labels=tf.zeros_like(self.z_logit_fake)
                    )
                    self.dis_loss_zz = tf.reduce_mean(self.loss_zz_real + self.loss_zz_fake)

        ############################################################################################
        # OPTIMIZERS
        ############################################################################################
        with tf.name_scope("Optimizers"):
            self.generator_optimizer = tf.train.AdamOptimizer(
                self.config.trainer.standard_lr_gen,
                beta1=self.config.trainer.optimizer_adam_beta1,
                beta2=self.config.trainer.optimizer_adam_beta2,
            )
            self.encoder_g_optimizer = tf.train.AdamOptimizer(
                self.config.trainer.standard_lr_enc,
                beta1=self.config.trainer.optimizer_adam_beta1,
                beta2=self.config.trainer.optimizer_adam_beta2,
            )
            self.encoder_r_optimizer = tf.train.AdamOptimizer(
                self.config.trainer.standard_lr_enc,
                beta1=self.config.trainer.optimizer_adam_beta1,
                beta2=self.config.trainer.optimizer_adam_beta2,
            )
            self.discriminator_optimizer = tf.train.AdamOptimizer(
                self.config.trainer.standard_lr_dis,
                beta1=self.config.trainer.optimizer_adam_beta1,
                beta2=self.config.trainer.optimizer_adam_beta2,
            )
            # Collect all the variables
            all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # Generator Network Variables
            self.generator_vars = [
                v for v in all_variables if v.name.startswith("SENCEBGAN/Generator_Model")
            ]
            # Discriminator Network Variables
            self.discriminator_vars = [
                v for v in all_variables if v.name.startswith("SENCEBGAN/Discriminator_Model")
            ]
            # Discriminator Network Variables
            self.encoder_g_vars = [
                v for v in all_variables if v.name.startswith("SENCEBGAN/Encoder_G_Model")
            ]
            self.encoder_r_vars = [
                v for v in all_variables if v.name.startswith("SENCEBGAN/Encoder_R_Model")
            ]
            self.dxxvars = [
                v for v in all_variables if v.name.startswith("SENCEBGAN/Discriminator_Model_XX")
            ]
            self.dzzvars = [
                v for v in all_variables if v.name.startswith("SENCEBGAN/Discriminator_Model_ZZ")
            ]
            # Generator Network Operations
            self.gen_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="SENCEBGAN/Generator_Model"
            )
            # Discriminator Network Operations
            self.disc_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="SENCEBGAN/Discriminator_Model"
            )
            self.encg_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="SENCEBGAN/Encoder_G_Model"
            )

            self.encr_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="SENCEBGAN/Encoder_R_Model"
            )
            self.update_ops_dis_xx = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="SENCEBGAN/Discriminator_Model_XX"
            )
            self.update_ops_dis_zz = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="SENCEBGAN/Discriminator_Model_ZZ"
            )
            with tf.control_dependencies(self.gen_update_ops):
                self.gen_op = self.generator_optimizer.minimize(
                    self.loss_generator,
                    var_list=self.generator_vars,
                    global_step=self.global_step_tensor,
                )
            with tf.control_dependencies(self.disc_update_ops):
                self.disc_op = self.discriminator_optimizer.minimize(
                    self.loss_discriminator, var_list=self.discriminator_vars
                )
            with tf.control_dependencies(self.encg_update_ops):
                self.encg_op = self.encoder_g_optimizer.minimize(
                    self.loss_encoder_g,
                    var_list=self.encoder_g_vars,
                    global_step=self.global_step_tensor,
                )
            with tf.control_dependencies(self.encr_update_ops):
                self.encr_op = self.encoder_r_optimizer.minimize(
                    self.loss_encoder_r,
                    var_list=self.encoder_r_vars,
                    global_step=self.global_step_tensor,
                )
            if self.config.trainer.enable_disc_xx:
                with tf.control_dependencies(self.update_ops_dis_xx):
                    self.disc_op_xx = self.discriminator_optimizer.minimize(
                        self.dis_loss_xx, var_list=self.dxxvars
                    )
            if self.config.trainer.enable_disc_zz:
                with tf.control_dependencies(self.update_ops_dis_zz):
                    self.disc_op_zz = self.discriminator_optimizer.minimize(
                        self.dis_loss_zz, var_list=self.dzzvars
                    )
            # if self.config.trainer.extra_gan_training:
            #     with tf.control_dependencies(self.gen_update_ops):
            #         self.gen_2_op = self.generator_optimizer.minimize(
            #             self.loss_generator_2, var_list=self.generator_vars
            #         )

            # Exponential Moving Average for Estimation
            self.dis_ema = tf.train.ExponentialMovingAverage(decay=self.config.trainer.ema_decay)
            maintain_averages_op_dis = self.dis_ema.apply(self.discriminator_vars)

            self.gen_ema = tf.train.ExponentialMovingAverage(decay=self.config.trainer.ema_decay)
            maintain_averages_op_gen = self.gen_ema.apply(self.generator_vars)
            # if self.config.trainer.extra_gan_training:
            #     self.gen_2_ema = tf.train.ExponentialMovingAverage(decay=self.config.trainer.ema_decay)
            #     maintain_averages_op_gen_2 = self.gen_2_ema.apply(self.generator_vars)

            self.encg_ema = tf.train.ExponentialMovingAverage(decay=self.config.trainer.ema_decay)
            maintain_averages_op_encg = self.encg_ema.apply(self.encoder_g_vars)

            self.encr_ema = tf.train.ExponentialMovingAverage(decay=self.config.trainer.ema_decay)
            maintain_averages_op_encr = self.encr_ema.apply(self.encoder_r_vars)

            if self.config.trainer.enable_disc_xx:
                self.dis_xx_ema = tf.train.ExponentialMovingAverage(
                    decay=self.config.trainer.ema_decay
                )
                maintain_averages_op_dis_xx = self.dis_xx_ema.apply(self.dxxvars)

            if self.config.trainer.enable_disc_zz:
                self.dis_zz_ema = tf.train.ExponentialMovingAverage(
                    decay=self.config.trainer.ema_decay
                )
                maintain_averages_op_dis_zz = self.dis_zz_ema.apply(self.dzzvars)

            with tf.control_dependencies([self.disc_op]):
                self.train_dis_op = tf.group(maintain_averages_op_dis)

            with tf.control_dependencies([self.gen_op]):
                self.train_gen_op = tf.group(maintain_averages_op_gen)

            # if self.config.trainer.extra_gan_training:
            #     with tf.control_dependencies([self.gen_2_op]):
            #         self.train_gen_op_2 = tf.group(maintain_averages_op_gen_2)

            with tf.control_dependencies([self.encg_op]):
                self.train_enc_g_op = tf.group(maintain_averages_op_encg)

            with tf.control_dependencies([self.encr_op]):
                self.train_enc_r_op = tf.group(maintain_averages_op_encr)

            if self.config.trainer.enable_disc_xx:
                with tf.control_dependencies([self.disc_op_xx]):
                    self.train_dis_op_xx = tf.group(maintain_averages_op_dis_xx)

            if self.config.trainer.enable_disc_zz:
                with tf.control_dependencies([self.disc_op_zz]):
                    self.train_dis_op_zz = tf.group(maintain_averages_op_dis_zz)

        ############################################################################################
        # TESTING
        ############################################################################################
        self.logger.info("Building Testing Graph...")
        with tf.variable_scope("SENCEBGAN"):
            with tf.variable_scope("Discriminator_Model"):
                self.embedding_q_ema, self.decoded_q_ema = self.discriminator(
                    self.image_input,
                    getter=get_getter(self.dis_ema),
                    do_spectral_norm=self.config.trainer.do_spectral_norm,
                )
            with tf.variable_scope("Generator_Model"):
                self.image_gen_ema = self.generator(
                    self.embedding_q_ema, getter=get_getter(self.gen_ema)
                )
            with tf.variable_scope("Discriminator_Model"):
                self.embedding_rec_ema, self.decoded_rec_ema = self.discriminator(
                    self.image_gen_ema,
                    getter=get_getter(self.dis_ema),
                    do_spectral_norm=self.config.trainer.do_spectral_norm,
                )
            # Second Training Part
            with tf.variable_scope("Encoder_G_Model"):
                self.image_encoded_ema = self.encoder_g(
                    self.image_input, getter=get_getter(self.encg_ema)
                )

            with tf.variable_scope("Generator_Model"):
                self.image_gen_enc_ema = self.generator(
                    self.image_encoded_ema, getter=get_getter(self.gen_ema)
                )
            with tf.variable_scope("Discriminator_Model"):
                self.embedding_enc_fake_ema, self.decoded_enc_fake_ema = self.discriminator(
                    self.image_gen_enc_ema,
                    getter=get_getter(self.dis_ema),
                    do_spectral_norm=self.config.trainer.do_spectral_norm,
                )
                self.embedding_enc_real_ema, self.decoded_enc_real_ema = self.discriminator(
                    self.image_input,
                    getter=get_getter(self.dis_ema),
                    do_spectral_norm=self.config.trainer.do_spectral_norm,
                )
            if self.config.trainer.enable_disc_xx:
                with tf.variable_scope("Discriminator_Model_XX"):
                    self.im_logit_real_ema, self.im_f_real_ema = self.discriminator_xx(
                        self.image_input,
                        self.image_input,
                        getter=get_getter(self.dis_xx_ema),
                        do_spectral_norm=self.config.trainer.do_spectral_norm,
                    )
                    self.im_logit_fake_ema, self.im_f_fake_ema = self.discriminator_xx(
                        self.image_input,
                        self.image_gen_enc_ema,
                        getter=get_getter(self.dis_xx_ema),
                        do_spectral_norm=self.config.trainer.do_spectral_norm,
                    )
            # Third training part
            with tf.variable_scope("Encoder_G_Model"):
                self.image_encoded_r_ema = self.encoder_g(self.image_input)

            with tf.variable_scope("Generator_Model"):
                self.image_gen_enc_r_ema = self.generator(self.image_encoded_r_ema)


            with tf.variable_scope("Encoder_R_Model"):
                self.image_ege_ema = self.encoder_r(self.image_gen_enc_r_ema)

            if self.config.trainer.enable_disc_zz:
                with tf.variable_scope("Discriminator_Model_ZZ"):
                    self.z_logit_real_ema, self.z_f_real_ema = self.discriminator_zz(
                        self.image_encoded_r_ema,
                        self.image_encoded_r_ema,
                        getter=get_getter(self.dis_zz_ema),
                        do_spectral_norm=self.config.trainer.do_spectral_norm,
                    )
                    self.z_logit_fake_ema, self.z_f_fake_ema = self.discriminator_zz(
                        self.image_encoded_r_ema,
                        self.image_ege_ema,
                        getter=get_getter(self.dis_zz_ema),
                        do_spectral_norm=self.config.trainer.do_spectral_norm,
                    )

        with tf.name_scope("Testing"):
            with tf.name_scope("Image_Based"):
                delta = self.image_input - self.image_gen_enc_ema
                self.mask = -delta
                delta_flat = tf.layers.Flatten()(delta)
                img_score_l1 = tf.norm(
                    delta_flat, ord=1, axis=1, keepdims=False, name="img_loss__1"
                )
                self.img_score_l1 = tf.squeeze(img_score_l1)

                delta = self.embedding_enc_fake_ema - self.embedding_enc_real_ema
                delta_flat = tf.layers.Flatten()(delta)
                img_score_l2 = tf.norm(
                    delta_flat, ord=1, axis=1, keepdims=False, name="img_loss__2"
                )
                self.img_score_l2 = tf.squeeze(img_score_l2)
                self.score_comb = (
                    (1 - self.config.trainer.feature_match_weight) * self.img_score_l1
                    + self.config.trainer.feature_match_weight * self.img_score_l2
                )
            with tf.name_scope("Noise_Based"):

                delta = self.image_encoded_r_ema - self.image_ege_ema
                delta_flat = tf.layers.Flatten()(delta)
                final_score_1 = tf.norm(
                    delta_flat, ord=1, axis=1, keepdims=False, name="final_score_1"
                )
                self.final_score_1 = tf.squeeze(final_score_1)

                delta = self.image_encoded_r_ema - self.image_ege_ema
                delta_flat = tf.layers.Flatten()(delta)
                final_score_2 = tf.norm(
                    delta_flat, ord=2, axis=1, keepdims=False, name="final_score_2"
                )
                self.final_score_2 = tf.squeeze(final_score_2)

                if self.config.trainer.enable_disc_xx:
                    # delta = self.im_logit_real_ema - self.im_logit_fake_ema
                    # delta_flat = tf.layers.Flattent()(delta)
                    # final_score_3 = tf.norm(delta_flat, ord=1, axis=1, keepdims=False, name="final_score_3")
                    # self.final_score_3 = tf.squeeze(final_score_3)

                    delta = self.im_f_real_ema - self.im_f_fake_ema
                    delta_flat = tf.layers.Flatten()(delta)
                    final_score_4 = tf.norm(
                        delta_flat, ord=1, axis=1, keepdims=False, name="final_score_4"
                    )
                    self.final_score_4 = tf.squeeze(final_score_4)

                if self.config.trainer.enable_disc_zz:
                    # delta = self.z_logit_real_ema - self.z_logit_fake_ema
                    # delta_flat = tf.layers.Flattent()(delta)
                    # final_score_5 = tf.norm(delta_flat, ord=1, axis=1, keepdims=False, name="final_score_5")
                    # self.final_score_5 = tf.squeeze(final_score_5)

                    delta = self.z_f_real_ema - self.z_f_fake_ema
                    delta_flat = tf.layers.Flatten()(delta)
                    final_score_6 = tf.norm(
                        delta_flat, ord=1, axis=1, keepdims=False, name="final_score_6"
                    )
                    self.final_score_6 = tf.squeeze(final_score_6)

        ############################################################################################
        # TENSORBOARD
        ############################################################################################
        if self.config.log.enable_summary:
            with tf.name_scope("train_summary"):
                with tf.name_scope("dis_summary"):
                    tf.summary.scalar("loss_disc", self.loss_discriminator, ["dis"])
                    tf.summary.scalar("loss_disc_real", self.disc_loss_real, ["dis"])
                    tf.summary.scalar("loss_disc_fake", self.disc_loss_fake, ["dis"])
                    if self.config.trainer.enable_disc_xx:
                        tf.summary.scalar("loss_dis_xx", self.dis_loss_xx, ["enc_g"])
                    if self.config.trainer.enable_disc_zz:
                        tf.summary.scalar("loss_dis_zz", self.dis_loss_zz, ["enc_r"])
                with tf.name_scope("gen_summary"):
                    tf.summary.scalar("loss_generator", self.loss_generator, ["gen"])
                with tf.name_scope("enc_summary"):
                    tf.summary.scalar("loss_encoder_g", self.loss_encoder_g, ["enc_g"])
                    tf.summary.scalar("loss_encoder_r", self.loss_encoder_r, ["enc_r"])
                with tf.name_scope("img_summary"):
                    tf.summary.image("input_image", self.image_input, 3, ["img_1"])
                    tf.summary.image("reconstructed", self.image_gen, 3, ["img_1"])
                    tf.summary.image("input_enc", self.image_input, 3, ["img_2"])
                    tf.summary.image("reconstructed", self.image_gen_enc, 3, ["img_2"])
                    tf.summary.image("input_image",self.image_input,1,["test"])
                    tf.summary.image("reconstructed", self.image_gen_enc_r_ema,1,["test"])
                    tf.summary.image("mask", self.mask, 1, ["test"])


            self.sum_op_dis = tf.summary.merge_all("dis")
            self.sum_op_gen = tf.summary.merge_all("gen")
            self.sum_op_enc_g = tf.summary.merge_all("enc_g")
            self.sum_op_enc_r = tf.summary.merge_all("enc_r")
            self.sum_op_im_1 = tf.summary.merge_all("img_1")
            self.sum_op_im_2 = tf.summary.merge_all("img_2")
            self.sum_op_test = tf.summary.merge_all("test")
            self.sum_op = tf.summary.merge([self.sum_op_dis, self.sum_op_gen])

    ###############################################################################################
    # MODULES
    ###############################################################################################
    def generator(self, noise_input, getter=None):
        with tf.variable_scope("Generator", custom_getter=getter, reuse=tf.AUTO_REUSE):
            net_name = "Layer_1"
            with tf.variable_scope(net_name):
                x_g = tf.layers.Dense(
                    units=2 * 2 * 256, kernel_initializer=self.init_kernel, name="fc"
                )(noise_input)
                x_g = tf.layers.batch_normalization(
                    x_g,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training_gen,
                    name="batch_normalization",
                )
                x_g = tf.nn.leaky_relu(
                    features=x_g, alpha=self.config.trainer.leakyReLU_alpha, name="relu"
                )
            x_g = tf.reshape(x_g, [-1, 2, 2, 256])
            net_name = "Layer_2"
            with tf.variable_scope(net_name):
                x_g = tf.layers.Conv2DTranspose(
                    filters=128,
                    kernel_size=5,
                    strides=2,
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv2t",
                )(x_g)
                x_g = tf.layers.batch_normalization(
                    x_g,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training_gen,
                    name="batch_normalization",
                )
                x_g = tf.nn.leaky_relu(
                    features=x_g, alpha=self.config.trainer.leakyReLU_alpha, name="relu"
                )
            net_name = "Layer_3"
            with tf.variable_scope(net_name):
                x_g = tf.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=5,
                    strides=2,
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv2t",
                )(x_g)
                x_g = tf.layers.batch_normalization(
                    x_g,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training_gen,
                    name="batch_normalization",
                )
                x_g = tf.nn.leaky_relu(
                    features=x_g, alpha=self.config.trainer.leakyReLU_alpha, name="relu"
                )
            net_name = "Layer_4"
            with tf.variable_scope(net_name):
                x_g = tf.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=5,
                    strides=2,
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv2t",
                )(x_g)
                x_g = tf.layers.batch_normalization(
                    x_g,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training_gen,
                    name="batch_normalization",
                )
                x_g = tf.nn.leaky_relu(
                    features=x_g, alpha=self.config.trainer.leakyReLU_alpha, name="relu"
                )
            net_name = "Layer_5"
            with tf.variable_scope(net_name):
                x_g = tf.layers.Conv2DTranspose(
                    filters=1,
                    kernel_size=5,
                    strides=2,
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv2t",
                )(x_g)
                x_g = tf.tanh(x_g, name="tanh")
        return x_g

    def discriminator(self, image_input, getter=None, do_spectral_norm=False):
        layers = sn if do_spectral_norm else tf.layers
        with tf.variable_scope("Discriminator", custom_getter=getter, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("Encoder"):
                x_e = tf.reshape(
                    image_input,
                    [-1, self.config.data_loader.image_size, self.config.data_loader.image_size, 1],
                )
                net_name = "Layer_1"
                with tf.variable_scope(net_name):
                    x_e = layers.conv2d(
                        x_e,
                        filters=32,
                        kernel_size=5,
                        strides=2,
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="conv",
                    )
                    x_e = tf.nn.leaky_relu(
                        features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                    )
                    # 14 x 14 x 64
                net_name = "Layer_2"
                with tf.variable_scope(net_name):
                    x_e = layers.conv2d(
                        x_e,
                        filters=64,
                        kernel_size=5,
                        padding="same",
                        strides=2,
                        kernel_initializer=self.init_kernel,
                        name="conv",
                    )
                    x_e = tf.layers.batch_normalization(
                        x_e,
                        momentum=self.config.trainer.batch_momentum,
                        training=self.is_training_dis,
                    )
                    x_e = tf.nn.leaky_relu(
                        features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                    )
                    # 7 x 7 x 128
                net_name = "Layer_3"
                with tf.variable_scope(net_name):
                    x_e = layers.conv2d(
                        x_e,
                        filters=128,
                        kernel_size=5,
                        padding="same",
                        strides=2,
                        kernel_initializer=self.init_kernel,
                        name="conv",
                    )
                    x_e = tf.layers.batch_normalization(
                        x_e,
                        momentum=self.config.trainer.batch_momentum,
                        training=self.is_training_dis,
                    )
                    x_e = tf.nn.leaky_relu(
                        features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                    )
                    # 4 x 4 x 256
                x_e = tf.layers.Flatten()(x_e)
                net_name = "Layer_4"
                with tf.variable_scope(net_name):
                    x_e = layers.dense(
                        x_e,
                        units=self.config.trainer.noise_dim,
                        kernel_initializer=self.init_kernel,
                        name="fc",
                    )

            embedding = x_e
            with tf.variable_scope("Decoder"):
                net = tf.reshape(embedding, [-1, 1, 1, self.config.trainer.noise_dim])
                net_name = "layer_1"
                with tf.variable_scope(net_name):
                    net = tf.layers.Conv2DTranspose(
                        filters=256,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="tconv1",
                    )(net)
                    net = tf.layers.batch_normalization(
                        inputs=net,
                        momentum=self.config.trainer.batch_momentum,
                        training=self.is_training_dis,
                        name="tconv1/bn",
                    )
                    net = tf.nn.relu(features=net, name="tconv1/relu")

                net_name = "layer_2"
                with tf.variable_scope(net_name):
                    net = tf.layers.Conv2DTranspose(
                        filters=128,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="tconv2",
                    )(net)
                    net = tf.layers.batch_normalization(
                        inputs=net,
                        momentum=self.config.trainer.batch_momentum,
                        training=self.is_training_dis,
                        name="tconv2/bn",
                    )
                    net = tf.nn.relu(features=net, name="tconv2/relu")

                net_name = "layer_3"
                with tf.variable_scope(net_name):
                    net = tf.layers.Conv2DTranspose(
                        filters=64,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="tconv3",
                    )(net)
                    net = tf.layers.batch_normalization(
                        inputs=net,
                        momentum=self.config.trainer.batch_momentum,
                        training=self.is_training_dis,
                        name="tconv3/bn",
                    )
                    net = tf.nn.relu(features=net, name="tconv3/relu")
                net_name = "layer_4"
                with tf.variable_scope(net_name):
                    net = tf.layers.Conv2DTranspose(
                        filters=32,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="tconv4",
                    )(net)
                    net = tf.layers.batch_normalization(
                        inputs=net,
                        momentum=self.config.trainer.batch_momentum,
                        training=self.is_training_dis,
                        name="tconv4/bn",
                    )
                    net = tf.nn.relu(features=net, name="tconv4/relu")
                net_name = "layer_5"
                with tf.variable_scope(net_name):
                    net = tf.layers.Conv2DTranspose(
                        filters=1,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="tconv5",
                    )(net)
                    decoded = tf.nn.tanh(net, name="tconv5/tanh")
        return embedding, decoded

    def encoder_g(self, image_input, getter=None):
        with tf.variable_scope("Encoder_G", custom_getter=getter, reuse=tf.AUTO_REUSE):
            x_e = tf.reshape(
                image_input,
                [-1, self.config.data_loader.image_size, self.config.data_loader.image_size, 1],
            )
            net_name = "Layer_1"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Conv2D(
                    filters=64,
                    kernel_size=5,
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )(x_e)
                x_e = tf.layers.batch_normalization(
                    x_e,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training_enc_g,
                )
                x_e = tf.nn.leaky_relu(
                    features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
            net_name = "Layer_2"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Conv2D(
                    filters=128,
                    kernel_size=5,
                    padding="same",
                    strides=(2, 2),
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )(x_e)
                x_e = tf.layers.batch_normalization(
                    x_e,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training_enc_g,
                )
                x_e = tf.nn.leaky_relu(
                    features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
            net_name = "Layer_3"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Conv2D(
                    filters=256,
                    kernel_size=5,
                    padding="same",
                    strides=(2, 2),
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )(x_e)
                x_e = tf.layers.batch_normalization(
                    x_e,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training_enc_g,
                )
                x_e = tf.nn.leaky_relu(
                    features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
            x_e = tf.layers.Flatten()(x_e)
            net_name = "Layer_4"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Dense(
                    units=self.config.trainer.noise_dim,
                    kernel_initializer=self.init_kernel,
                    name="fc",
                )(x_e)
        return x_e

    def encoder_r(self, image_input, getter=None):
        with tf.variable_scope("Encoder_R", custom_getter=getter, reuse=tf.AUTO_REUSE):
            x_e = tf.reshape(
                image_input,
                [-1, self.config.data_loader.image_size, self.config.data_loader.image_size, 1],
            )
            net_name = "Layer_1"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Conv2D(
                    filters=64,
                    kernel_size=5,
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )(x_e)
                x_e = tf.layers.batch_normalization(
                    x_e,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training_enc_r,
                )
                x_e = tf.nn.leaky_relu(
                    features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
            net_name = "Layer_2"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Conv2D(
                    filters=128,
                    kernel_size=5,
                    padding="same",
                    strides=(2, 2),
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )(x_e)
                x_e = tf.layers.batch_normalization(
                    x_e,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training_enc_r,
                )
                x_e = tf.nn.leaky_relu(
                    features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
            net_name = "Layer_3"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Conv2D(
                    filters=256,
                    kernel_size=5,
                    padding="same",
                    strides=(2, 2),
                    kernel_initializer=self.init_kernel,
                    name="conv",
                )(x_e)
                x_e = tf.layers.batch_normalization(
                    x_e,
                    momentum=self.config.trainer.batch_momentum,
                    training=self.is_training_enc_r,
                )
                x_e = tf.nn.leaky_relu(
                    features=x_e, alpha=self.config.trainer.leakyReLU_alpha, name="leaky_relu"
                )
            x_e = tf.layers.Flatten()(x_e)
            net_name = "Layer_4"
            with tf.variable_scope(net_name):
                x_e = tf.layers.Dense(
                    units=self.config.trainer.noise_dim,
                    kernel_initializer=self.init_kernel,
                    name="fc",
                )(x_e)
        return x_e

    # Regularizer discriminator for the Generator Encoder
    def discriminator_xx(self, img_tensor, recreated_img, getter=None, do_spectral_norm=False):
        """ Discriminator architecture in tensorflow

        Discriminates between (x, x) and (x, rec_x)
        Args:
            img_tensor:
            recreated_img:
            getter: for exponential moving average during inference
            reuse: sharing variables or not
            do_spectral_norm:
        """
        layers = sn if do_spectral_norm else tf.layers
        with tf.variable_scope("Discriminator_xx", reuse=tf.AUTO_REUSE, custom_getter=getter):
            net = tf.concat([img_tensor, recreated_img], axis=1)
            net_name = "layer_1"
            with tf.variable_scope(net_name):
                net = layers.conv2d(
                    net,
                    filters=64,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv1",
                )
                net = tf.nn.leaky_relu(
                    features=net, alpha=self.config.trainer.leakyReLU_alpha, name="conv2/leaky_relu"
                )
                net = tf.layers.dropout(
                    net,
                    rate=self.config.trainer.dropout_rate,
                    training=self.is_training_enc_g,
                    name="dropout",
                )
            with tf.variable_scope(net_name, reuse=True):
                weights = tf.get_variable("conv1/kernel")

            net_name = "layer_2"
            with tf.variable_scope(net_name):
                net = layers.conv2d(
                    net,
                    filters=128,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=self.init_kernel,
                    name="conv2",
                )
                net = tf.nn.leaky_relu(
                    features=net, alpha=self.config.trainer.leakyReLU_alpha, name="conv2/leaky_relu"
                )
                net = tf.layers.dropout(
                    net,
                    rate=self.config.trainer.dropout_rate,
                    training=self.is_training_enc_g,
                    name="dropout",
                )
            net = tf.layers.Flatten()(net)

            intermediate_layer = net

            net_name = "layer_3"
            with tf.variable_scope(net_name):
                net = tf.layers.dense(net, units=1, kernel_initializer=self.init_kernel, name="fc")
                logits = tf.squeeze(net)

        return logits, intermediate_layer

    # Regularizer discriminator for the Reconstruction Encoder

    def discriminator_zz(self, noise_tensor, recreated_noise, getter=None, do_spectral_norm=False):
        """ Discriminator architecture in tensorflow

        Discriminates between (z, z) and (z, rec_z)
        Args:
            noise_tensor:
            recreated_noise:
            getter: for exponential moving average during inference
            reuse: sharing variables or not
            do_spectral_norm:
        """
        layers = sn if do_spectral_norm else tf.layers

        with tf.variable_scope("Discriminator_zz", reuse=tf.AUTO_REUSE, custom_getter=getter):
            y = tf.concat([noise_tensor, recreated_noise], axis=-1)

            net_name = "y_layer_1"
            with tf.variable_scope(net_name):
                y = layers.dense(y, units=64, kernel_initializer=self.init_kernel, name="fc")
                y = tf.nn.leaky_relu(features=y, alpha=self.config.trainer.leakyReLU_alpha)
                y = tf.layers.dropout(
                    y,
                    rate=self.config.trainer.dropout_rate,
                    training=self.is_training_enc_r,
                    name="dropout",
                )

            net_name = "y_layer_2"
            with tf.variable_scope(net_name):
                y = layers.dense(y, units=32, kernel_initializer=self.init_kernel, name="fc")
                y = tf.nn.leaky_relu(features=y, alpha=self.config.trainer.leakyReLU_alpha)
                y = tf.layers.dropout(
                    y,
                    rate=self.config.trainer.dropout_rate,
                    training=self.is_training_enc_r,
                    name="dropout",
                )

            intermediate_layer = y

            net_name = "y_layer_3"
            with tf.variable_scope(net_name):
                y = layers.dense(y, units=1, kernel_initializer=self.init_kernel, name="fc")
                logits = tf.squeeze(y)

        return logits, intermediate_layer

    ###############################################################################################
    # CUSTOM LOSSES
    ###############################################################################################
    def mse_loss(self, pred, data, mode="norm", order=2):
        if mode == "norm":
            delta = pred - data
            delta = tf.layers.Flatten()(delta)
            loss_val = tf.norm(delta, ord=order, axis=1, keepdims=False)
        elif mode == "mse":
            loss_val = tf.reduce_mean(tf.squared_difference(pred, data))
        return loss_val

    def pullaway_loss(self, embeddings):
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm
        similarity = tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)
        batch_size = tf.cast(tf.shape(embeddings)[0], tf.float32)
        pt_loss = (tf.reduce_sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
        return pt_loss

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.log.max_to_keep)
