import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from utils.utils import get_args
from utils.config import process_config


from mains.main_gan import main
from mains.main_gan_mark2 import main as main_mark2

def run():
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)
    
    if (config.model_class == "gan"):
        main(config)
    if (config.model_class == "gan_mark2"):
        main_mark2(config)


if __name__ == '__main__':
    run()
