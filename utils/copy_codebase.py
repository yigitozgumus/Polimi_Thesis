from shutil import copy


def copy_codebase(config):
    # Get class name
    model_name = config.model.name.split(".")[0]
    model_src_file = "models/{}.py".format(model_name)
    trainer_name = config.trainer.name.split(".")[0]
    trainer_src_file = "trainers/{}.py".format(trainer_name)
    config_file = "configs/{}.json".format(model_name)

    copy(src=model_src_file, dst=config.log.codebase_dir)
    copy(src=trainer_src_file, dst=config.log.codebase_dir)
    copy(src=config_file, dst=config.log.codebase_dir)
