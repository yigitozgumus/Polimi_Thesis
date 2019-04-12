from utils.utils import get_args
from utils.config import process_config

from mains.main_gan_tf import main
from mains.main_gan_eager import main as main_eager
from mains.main_gan_keras import main as main_keras
from mains.main_gan_tf import main as main_tf
from mains.main_alad import main as main_alad
from mains.main_anogan import main as main_anogan

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def run():
    try:
        args = get_args()
        config = process_config(args.config, args.experiment)
    except:
        print("missing or invalid arguments")
        exit(0)

    if config.model.name == "gan":
        main(config)

    if config.model.name == "anogan":
        main_anogan(config)

    elif config.model.name == "gan_eager":
        main_eager(config)

    elif config.model.name == "gan_keras":
        main_keras(config)

    elif config.model.name == "gan_tf":
        main_tf(config)

    elif config.model.name == "alad":
        main_alad(config)


if __name__ == "__main__":
    run()
