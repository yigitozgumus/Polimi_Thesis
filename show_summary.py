import sys
import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from utils.config import process_config
from models.gan_model import GAN
from models.gan_model_mark2 import GAN_mark2


def main():
    config = process_config(sys.argv[1])
    if config.model_class == "gan":
        model = GAN(config)
    elif config.model_class == "gan_mark2":
        model = GAN_mark2(config)
    print("Generator Network")
    model.generator.summary()
    print("Discriminator Network")
    model.discriminator.summary()


if __name__ == "__main__":
    main()
