from utils.utils import get_args
from utils.config import process_config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from mains.main_gan import main

def run():
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)
    
    main(config)


if __name__ == '__main__':
    run()
