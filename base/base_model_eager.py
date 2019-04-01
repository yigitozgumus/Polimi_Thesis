import tensorflow as tf
import tensorflow.contrib.eager as tfe
#tf.enable_eager_execution()

class BaseModelEager:
    def __init__(self, config):
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self):
        print("Saving model...")
        self.checkpoint.save(self.config.checkpoint_prefix)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.checkpoint.restore(latest_checkpoint)
            print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tfe.Variable(0, name='cur_epoch')

    def increment_cur_epoch_tensor(self):
        with tf.variable_scope('cur_epoch'):
            tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tfe.Variable(
                0, name='global_step')

    def init_saver(self):
        # just copy the following line in your child class
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

