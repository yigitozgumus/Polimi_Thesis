import tensorflow as tf


from data_loader.data_generator_keras import DataGeneratorKeras
from models.gankeras import GanKeras
from trainers.gan_trainer_keras import GANTrainerKeras

from utils.dirs import create_dirs


def main(config):
    # capture the config path from the run arguments
    # then process the json configuration file

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir,
                 config.step_generation_dir])
    # Create a session
    sess = tf.Session()
    # create your data generator
    data = DataGeneratorKeras(config)
    # create an instance of the model you want
    model = GanKeras(config)
    # create trainer and pass all the previous components to it
    trainer = GANTrainerKeras(sess,model, data, config)
    # load model if exists
    #model.load()
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
