import tensorflow as tf

from utils.data_generator import DataGenerator
from models.gan_model import GAN
from trainers.gan_trainer import GANTrainer
from utils.config import process_config
from utils.logger import Logger
from utils.utils import get_args
from utils.dirs import create_dirs


def main(config):
    # capture the config path from the run arguments
    # then process the json configuration file

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    data = DataGenerator(config)
    iterator = data.dataset.make_initializable_iterator()
    # create an instance of the model you want
    model = GAN(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = GANTrainer(sess, model, iterator, config, logger)
    #load model if exists
    model.load(sess)
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()

