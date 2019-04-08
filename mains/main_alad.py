import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.alad_tf import ALAD_TF
from trainers.alad_trainer import ALAD_Trainer

from utils.summarizer import Summarizer

from utils.dirs import create_dirs


def main(config):
    # capture the config path from the run arguments
    # then process the json configuration file

    # create the experiments dirs
    create_dirs([config.log.summary_dir,
                 config.log.checkpoint_dir,
                 config.log.step_generation_dir,
                 config.log.log_file_dir])
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


if __name__ == '__main__':
    main()
