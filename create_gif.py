import imageio
import os
import argparse
import sys
from utils.dirs import listdir_nohidden
from utils.config import process_config


def create_gif(folder_name, output_name):
    files = listdir_nohidden(folder_name)
    num_epochs = len(files)
    images = []
    for filename in files:
        images.append(imageio.imread(folder_name + "/" + filename))
    imageio.mimsave(output_name +"_"+str(num_epochs)+ ".gif", images)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("-c", "--config", metavar="C", help="The Configuration file")
    argparser.add_argument("-n", "--name", metavar="N", help="Name of the gif")
    args = argparser.parse_args()

    config = process_config(args.config)
    folder = os.path.join("Logs", config.exp_name, "generated")
    create_gif(folder, args.name)


if __name__ == "__main__":
    main()
