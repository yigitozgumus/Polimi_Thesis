import os
import sys
import tensorflow as tf
from utils.config import process_config
from models.gan_model import GAN
def main():
    config = process_config(sys.argv[1])

    model = GAN(config)
    print("Generator Network")
    model.generator.summary()
    print("Discriminator Network")
    model.discriminator.summary()

if __name__ == "__main__":
    main()