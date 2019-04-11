import tensorflow as tf

from data_loader.data_generator_eager import DataGeneratorEager
from models.gan_eager import GAN_eager
from trainers.gan_trainer_eager import GANTrainerEager
from utils.summarizer_eager import Summarizer_eager
from utils.dirs import create_dirs
from utils.logger import Logger


def main(config):
    create_dirs(
        [
            config.log.summary_dir,
            config.log.checkpoint_dir,
            config.log.step_generation_dir,
            config.log.log_file_dir,
        ]
    )
    l = Logger(config)
    logger = l.get_logger(__name__)

    # create tensorflow session
    logger.info("Experiment has begun")
    # create your data generator
    data = DataGeneratorEager(config)
    # create an instance of the model you want
    model = GAN_eager(config)
    # create tensorboard logger
    summarizer = Summarizer_eager(config)
    # create trainer and pass all the previous components to it
    trainer = GANTrainerEager(model, data, config, summarizer)
    # load model if exists
    model.load()
    # here you train your model
    trainer.train()


if __name__ == "__main__":
    main()
