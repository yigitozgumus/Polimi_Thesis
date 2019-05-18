import json
from dotmap import DotMap
import os


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, "r") as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using Namespace
    config = DotMap(config_dict)

    return config, config_dict


def create_parameter_file(config: object) -> None:
    location = config.log.parameter_dir
    # If the folder is present then already the experiment values are created
    if not os.path.exists(location):
        os.makedirs(location)
        name = "parameters.txt"
        f = open(os.path.join(location, name), "w+")
        f.write("{}\n".format("-" * 70))
        for category, sub_list in config.items():
            f.write("{}:\n".format(category))
            for element in sub_list.items():
                f.write("  {} : {}\n".format(element[0], element[1]))
            f.write("{}\n".format("-" * 70))
        f.close()


def process_config(config) -> object:

    config.log.summary_dir = os.path.join(config.log.output_folder, config.exp.name, "summary/")
    config.log.checkpoint_dir = os.path.join(
        config.log.output_folder, config.exp.name, "checkpoint/"
    )
    config.log.log_file_dir = os.path.join(config.log.output_folder, config.exp.name, "logs/")
    config.log.checkpoint_prefix = os.path.join(config.log.checkpoint_dir, "ckpt")
    config.log.step_generation_dir = os.path.join(
        config.log.output_folder, config.exp.name, "generated/"
    )
    config.log.parameter_dir = os.path.join(
        config.log.output_folder, config.exp.name, "parameters/"
    )
    config.log.result_dir = os.path.join(config.log.output_folder, config.exp.name, "results/")
    config.log.codebase_dir = os.path.join(config.log.output_folder, config.exp.name, "codebase/")
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
