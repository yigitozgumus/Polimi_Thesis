import tensorflow as tf
from utils.utils import get_args
from utils.config import process_config
from utils.config import get_config_from_json
from utils.factory import create
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.copy_codebase import copy_codebase
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def run_multi():
    # Get the arguments
    args = get_args()
    config, _ = get_config_from_json(args.config)
    values_sn = config.exp.vals_0
    values_train = config.exp.vals_1
    values_init = config.exp.vals_2
    params = config.exp.params
    section = config.exp.section
    # Spectral Normalization
    for i in values_sn:
    # Mode
        for j in values_train:
        # Init
            for k in values_init:
                config[section][params[0]] = i
                config[section][params[1]] = j
                config[section][params[2]] = k
                config.exp.name = args.experiment + "_{}_{}_{}".format(i, j, k)
                process_config(config)
                create_dirs(
                    [
                        config.log.summary_dir,
                        config.log.checkpoint_dir,
                        config.log.step_generation_dir,
                        config.log.log_file_dir,
                        config.log.codebase_dir,
                    ]
                )
                # Copy the model code and the trainer code to the experiment folder
                run(config, args)
                tf.reset_default_graph()
                # Delete the session and the model


def run(config):
    copy_codebase(config)

    l = Logger(config)
    logger = l.get_logger(__name__)
    # Set the random seed
    tf.random.set_random_seed(config.data_loader.random_seed)
    # Create the tensorflow session
    sess = tf.Session()
    # Create the dataloader
    data = create("data_loader." + config.data_loader.name)(config)
    # Create the model instance
    model = create("models.32." + config.model.name)(config)
    # Create the summarizer Object
    summarizer = create("utils." + config.log.name)(sess, config)
    # Create the trainer
    trainer = create("trainers." + config.trainer.name)(sess, model, data, config, summarizer)
    # Load model if exists
    model.load(sess)
    # Train the model
    trainer.train()
    # Test the model
    if config.trainer.test_at_end:
        trainer.test()
    logger.info("Experiment has ended.")


if __name__ == "__main__":
    run_multi()
