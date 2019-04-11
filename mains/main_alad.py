import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.alad_tf import ALAD_TF
from trainers.alad_trainer import ALAD_Trainer
from utils.summarizer import Summarizer
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
            config.log.result_dir,
        ]
    )
    l = Logger(config)
    logger = l.get_logger(__name__)
    logger.info("Experiment has begun")
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    data = DataGenerator(config)
    # create an instance of the model you want
    model = ALAD_TF(config)
    # create tensorboard logger
    summarizer = Summarizer(sess, config)
    # create trainer and pass all the previous components to it
    trainer = ALAD_Trainer(sess, model, data, config, summarizer)
    # load model if exists
    model.load(sess)
    # here you train your model
    trainer.train()
    logger.info("Experiment has ended.")


if __name__ == "__main__":
    main()
