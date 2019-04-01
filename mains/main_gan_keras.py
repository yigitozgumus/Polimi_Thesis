import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.gankeras import GanKeras
from trainers.gan_trainer_keras import GANTrainerKeras
from utils.logger import Logger
from utils.dirs import create_dirs


def main(config):
    # capture the config path from the run arguments
    # then process the json configuration file

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir,
                 config.step_generation_dir])
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    data = DataGenerator(config)
    # create an instance of the model you want
    model = GanKeras(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = GANTrainerKeras(sess, model, data, config, logger)
    # load model if exists
    model.load(sess)
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
