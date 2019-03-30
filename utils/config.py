
import json
from argparse import Namespace
import os


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using Namespace
    config = Namespace(**config_dict)

    return config, config_dict


def create_parameter_file(config: object) -> None:
    location = config.parameter_dir
    # If the folder is present then already the experiment values are created
    if not os.path.exists(location):
        os.makedirs(location)
        name = "parameters.txt"
        f = open(os.path.join(location,name),"w+")
        f.write("----------------------------------------------------------\n")
        f.write("Experiment: {}\n".format(config.exp_name))
        f.write("----------------------------------------------------------\n")
        parameters = config._get_kwargs()
        for param in parameters:
            f.write("{} : {}\n".format(param[0],param[1]))
        f.close()
    else:
        print("Experiment parameters are already stored")


def process_config(json_file: str, exp_name: str) -> object:
    config, _ = get_config_from_json(json_file)
    config.exp_name = exp_name
    config.summary_dir = os.path.join(config.output_folder, config.exp_name, "summary/")
    config.checkpoint_dir = os.path.join(config.output_folder, config.exp_name, "checkpoint/")
    config.checkpoint_prefix = os.path.join(config.checkpoint_dir, "ckpt")
    config.step_generation_dir = os.path.join(config.output_folder, config.exp_name, "generated/")
    config.parameter_dir = os.path.join(config.output_folder,config.exp_name, "parameters/")
    create_parameter_file(config)
    return config


def create_dirs(dirs: list) -> None:
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


