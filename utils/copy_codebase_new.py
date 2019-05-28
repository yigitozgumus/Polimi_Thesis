from shutil import copy


def copy_codebase(config):
    # Get class name
    model_name = config.model.name.split(".")[0]
    image_size = config.data_loader.image_size
    model_src_file = "models/new/{}.py".format(model_name)
    trainer_name = config.trainer.name.split(".")[0]
    trainer_src_file = "trainers/{}.py".format(trainer_name)
    config_file = "configs/new/{}.json".format(model_name)

    copy(src=model_src_file, dst=config.log.codebase_dir)
    copy(src=trainer_src_file, dst=config.log.codebase_dir)
    copy(src=config_file, dst=config.log.codebase_dir)
