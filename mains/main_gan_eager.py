import tensorflow as tf

from data_loader.data_generator_eager import DataGeneratorEager
from models.gan_eager import GAN_eager
from trainers.gan_trainer_eager import GANTrainer_eager
from utils.logger_eager import Logger_eager
from utils.dirs import create_dirs


def main(config):
    # capture the config path from the run arguments
    # then process the json configuration file

    # create the experiments dirs
    create_dirs(
        [config.summary_dir, config.checkpoint_dir,config.step_generation_dir])
    # create your data generator
    data = DataGeneratorEager(config)
    # create an instance of the model you want
    model = GAN_eager(config)
    # create tensorboard logger
    logger = Logger_eager(config)
    # create trainer and pass all the previous components to it
    trainer = GANTrainer_eager(model, data, config, logger)
    #load model if exists
    model.load()
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()

