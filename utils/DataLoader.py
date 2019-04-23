import os
import numpy as np
from skimage import io
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from utils.utils import working_directory
from utils.download_data import download_data
from utils.dirs import listdir_nohidden
from utils.logger import Logger


class DataLoader:
    def __init__(self, config):
        """
        Args:
            data_dir: this folder path should contain both Anomalous and Normal images
        """
        self.config = config

        log_object = Logger(self.config)
        self.logger = log_object.get_logger(__name__)
        self.data_dir = self.config.dirs.data
        self.train_dataset = os.path.join(self.data_dir, "train")
        self.valid_dataset = os.path.join(self.data_dir, "valid")
        self.img_location = os.path.join(self.data_dir, "test", "imgs/")
        self.tag_location = os.path.join(self.data_dir, "test", "labels/")
        if not os.path.exists(self.data_dir):
            self.logger.info("Dataset is not present. Download is started.")
            download_data(self.data_dir)
        self.data_dir_normal = self.config.dirs.data_normal
        self.data_dir_anomalous = self.config.dirs.data_anomalous
        # Up until this part only the raw dataset existence is checked and downloaded if not
        self.dataset_name = None
        # this is to list all the folders
        self.dir_names = listdir_nohidden(self.data_dir)
        self.test_size_per_img = (
            None
        )  # This will be the number of patches that will be extracted from each test image
        # Normal images for the train and validation dataset
        normal_imgs = self.data_dir_normal
        # Anormal images and the tag infor regarding the anomaly for test set
        anorm_imgs = self.data_dir_anomalous + "/images/"
        anorm_tag_imgs = self.data_dir_anomalous + "/gt/"
        norm_img_names = [normal_imgs + x for x in listdir_nohidden(normal_imgs)]
        anorm_img_names = [anorm_imgs + x for x in listdir_nohidden(anorm_imgs)]
        anorm_tag_names = [anorm_tag_imgs + x for x in listdir_nohidden(anorm_tag_imgs)]
        self.norm_img_array = self.create_image_array(norm_img_names, save=False)
        self.anorm_img_array = self.create_image_array(anorm_img_names, save=False)
        self.anorm_tag_array = self.create_image_array(anorm_tag_names, save=False)
        self.image_tag_list = list(zip(self.anorm_img_array, self.anorm_tag_array))
        if not self.config.data_loader.validation:
            self.populate_train()
        else:
            self.populate_train_valid()
        if self.config.data_loader.mode == "anomaly":
            self.populate_test()

    def populate_train(self):
        # Check if we have the data already
        if "train" in self.dir_names:
            self.logger.info("Train Dataset is already populated.")
        else:
            self.logger.info("Train Dataset will be populated")
            size = self.config.data_loader.image_size
            num_images = 10240
            imgs = []
            for ind, img in enumerate(self.norm_img_array):
                h, w = img.shape[:2]
                new_h, new_w = size, size
                for idx in range(num_images):
                    top = np.random.randint(0, h - new_h)
                    left = np.random.randint(0, w - new_w)
                    image = img[top : top + new_h, left : left + new_w]
                    imgs.append(image)
                self.logger.debug("{} images generated".format(num_images * (ind + 1)))
            # Check if the folder is there
            if not os.path.exists(self.train_dataset):
                os.mkdir(self.train_dataset)
            with working_directory(self.train_dataset):
                for idx, img in enumerate(imgs):
                    im = Image.fromarray(img)
                    im.save("img_{}.jpg".format(str(idx)))

    def populate_train_valid(self):
        if "train" in self.dir_names and "valid" in self.dir_names:
            self.logger.info("Train and Validation datasets are already populated")
        else:
            # Remove train dataset from the previous run
            os.rmdir(self.train_dataset)
            self.logger.info("Train and Validations Datasets will be populated")
            size = self.config.data_loader.image_size
            num_images = 10240
            imgs = []
            for ind, img in enumerate(self.norm_img_array):
                h, w = img.shape[:2]
                new_h, new_w = size, size
                for idx in range(num_images):
                    top = np.random.randint(0, h - new_h)
                    left = np.random.randint(0, w - new_w)
                    image = img[top : top + new_h, left : left + new_w]
                    imgs.append(image)
                self.logger.debug("{} images generated".format(num_images * (ind + 1)))
            # Creation of validation dataset
            np.random.seed(self.config.data_loader.random_seed)
            validation_list = np.random.choice(51200, 5120)  # 10% of the training set
            imgs_train = [x for ind, x in enumerate(imgs) if ind not in validation_list]
            imgs_valid = [x for ind, x in enumerate(imgs) if ind in validation_list]
            # Check if the folder is there
            if not os.path.exists(self.train_dataset):
                os.mkdir(self.train_dataset)
            # Check if the folder is there
            if not os.path.exists(self.valid_dataset):
                os.mkdir(self.valid_dataset)
            with working_directory(self.train_dataset):
                for idx, img in enumerate(imgs_train):
                    im = Image.fromarray(img)
                    im.save("img_{}.jpg".format(str(idx)))
            with working_directory(self.valid_dataset):
                for idx, img in enumerate(imgs_valid):
                    im = Image.fromarray(img)
                    im.save("img_{}.jpg".format(str(idx)))

    def populate_test(self):
        if "test" in self.dir_names:
            self.logger.info("Test Dataset is already populated")
        else:
            self.logger.info("Test Dataset will be populated")
            size = self.config.data_loader.image_size
            folder_name = "test"
            first_level = os.path.join(self.data_dir, folder_name)
            if not os.path.exists(first_level):
                os.mkdir(first_level)
            img_files = []
            tag_files = []
            for img_, tag_ in self.image_tag_list:
                h, w = img_.shape[:2]
                self.w_turns = w // size
                self.h_turns = h // size

                for adv_h in range(self.h_turns):
                    for adv_w in range(self.w_turns):
                        image = img_[
                            adv_h * size : (adv_h + 1) * size, adv_w * size : (adv_w + 1) * size
                        ]
                        tag = tag_[
                            adv_h * size : (adv_h + 1) * size, adv_w * size : (adv_w + 1) * size
                        ]
                        img_files.append(image)
                        tag_files.append(tag)
            self.test_size_per_img = self.w_turns * self.h_turns
            if not os.path.exists(self.img_location):
                os.mkdir(self.img_location)
            with working_directory(self.img_location):
                for idx, img in enumerate(img_files):
                    im = Image.fromarray(img)
                    im.save(
                        "img_{}_{}.jpg".format(
                            idx // self.test_size_per_img, idx % self.test_size_per_img
                        )
                    )
            if not os.path.exists(self.tag_location):
                os.mkdir(self.tag_location)
            with working_directory(self.tag_location):
                for idx, tag in enumerate(tag_files):
                    im = Image.fromarray(tag)
                    im.save(
                        "label_{}_{}.jpg".format(
                            idx // self.test_size_per_img, idx % self.test_size_per_img
                        )
                    )

    def create_image_array(self, img_names, save=True, file_name="Dataset"):
        """
        Args:
            img_names:
        """
        self.dataset_name = os.path.join(self.data_dir, file_name)
        img_array = []
        for img in img_names:
            im2arr = io.imread(img)
            img_array.append(im2arr)
        if save:
            np.save(self.dataset_name, img_array)
        return np.array(img_array)

    def get_train_dataset(self):
        """
        :param size: size of the image
        :return: numpy array of images and corresponding labels
        """
        img_list = listdir_nohidden(self.train_dataset)
        img_names = tf.constant([os.path.join(self.train_dataset, x) for x in img_list])
        self.logger.info("Train Dataset is Loaded")
        return img_names

    def get_valid_dataset(self):
        """
        :param size: size of the image
        :return: numpy array of images and corresponding labels
        """
        img_list = listdir_nohidden(self.valid_dataset)
        img_names = tf.constant([os.path.join(self.train_dataset, x) for x in img_list])
        self.logger.info("Validation Dataset is Loaded")
        return img_names

    def get_test_dataset(self):
        """
        :param size: size of the image
        :return: numpy array of images and corresponding labels
        """
        img_list = listdir_nohidden(self.img_location)
        img_names = tf.constant([os.path.join(self.img_location, x) for x in img_list])
        tag_list = listdir_nohidden(self.tag_location)
        tag_list_merged = [os.path.join(self.tag_location, x) for x in tag_list]
        labels = []
        for label in tag_list_merged:
            im2arr = io.imread(label)
            labels.append(1) if np.sum(im2arr) else labels.append(0)
        labels_f = tf.constant(labels)

        return [img_names, labels_f]
