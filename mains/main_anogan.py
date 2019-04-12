import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.anogan import ANOGAN
from trainers.anogan_trainer import ANOGAN_Trainer
from utils.config import process_config
from utils.summarizer import Summarizer
from utils.utils import get_args
from utils.dirs import create_dirs
from utils.logger import Logger


def main(config):
    # create the experiments dirs
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
    sess = tf.Session()
    data = DataGenerator(config)
    # create an instance of the model you want
    model = ANOGAN(config)
    # create the Summarizer object
    summarizer = Summarizer(sess, config)
    # create your data generator
    # create trainer and pass all the previous components to it
    trainer = ANOGAN_Trainer(sess, model, data, config, summarizer)
    # load model if exists
    model.load(sess)
    # here you train your model
    trainer.train()
    logger.info("Experiment has ended.")


if __name__ == "__main__":
    main()
