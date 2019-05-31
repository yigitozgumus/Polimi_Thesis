import os
from contextlib import contextmanager
import json
import argparse
from argparse import Namespace


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Namespace(**config_dict)

    return config, config_dict



# function for changing directory
@contextmanager
def working_directory(directory):
    owd = os.getcwd()
    try:
        os.chdir(directory)
        yield directory
    finally:
        os.chdir(owd)


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', dest='config',
        help='The Configuration file')
    argparser.add_argument(
        '-e', dest='experiment',
        help='Experiment name of the model')
    argparser.add_argument('--train', dest='train', action='store_true')
    argparser.add_argument('--test', dest='train', action='store_false')
    argparser.set_defaults(train=True)

    args = argparser.parse_args()
    return args
