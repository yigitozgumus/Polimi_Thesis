from utils.utils import get_args
from utils.config import process_config

from mains.main_gan import main
from mains.main_gan_mark2 import main as main_mark2
from mains.main_gan_eager import main as main_eager
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def run():
    try:
        args = get_args()
        config = process_config(args.config, args.experiment)
    except:
        print("missing or invalid arguments")
        exit(0)

    if (config.model_class == "gan"):
        main(config)
    elif (config.model_class == "gan_mark2"):
        main_mark2(config)
    elif (config.model_class == "gan_eager"):
        main_eager(config)

if __name__ == '__main__':
    run()
