import os
import argparse
import subprocess
from utils.dirs import listdir_nohidden
from utils.utils import working_directory

# This function will check if the training process is running. If it's running then that is fine.
# If it's not running, it will check the total number of generated epoch images from the generated
# Folder by extracting the total number of epochs and the image generation frequency. If the number of
# images are not complete, it will re-start the experiment
def check_training_process(args):
    # Check if the process is running or not
    try:
        p = subprocess.run('ps aux | grep "[p]ython run.py"',shell=True,check=True)
    except subprocess.CalledProcessError as e:
        print("TrainerChecker: Process is not working")
        p = None
    if p is not None:
        # process is running, leave it alone
        pass
    else:
        # Process is not running, so we need to check the images
        parameter_dir = os.path.join("Logs", args.experiment, "parameters")
        with working_directory(parameter_dir):
            f = open("parameters.txt","r")
            lines = f.readlines()
        print("Getting the info")
        epochs = int(lines[21].split(":")[1].strip("\n "))
        print(epochs)
        epochs_per_image = int(lines[22].split(":")[1].strip("\n "))
        print(epochs_per_image)
            # This is the number of images we need to find
        num_finished = epochs // epochs_per_image + 2
        print("We need {} images".format(num_finished))
        image_dir = os.path.join("Logs",args.experiment,"generated")
        with working_directory(image_dir):
            image_num = len(listdir_nohidden("."))

        if (image_num != num_finished):
            program_command = "python run.py -c {} -e {}".format(args.config,args.experiment)
            subprocess.run(program_command,shell=True)

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("-c","--config", metavar="C", help="Required config file if the process needs to be restarted")
    argparser.add_argument("-e","--experiment",metavar="E", help="Experiment Name")
    args = argparser.parse_args()
    check_training_process(args)


if __name__ == "__main__":
    main()