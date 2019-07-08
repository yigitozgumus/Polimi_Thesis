import os
import numpy as np
from skimage import io
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from utils.utils import working_directory
from utils.download_data import download_data_material
from utils.dirs import listdir_nohidden
from utils.logger import Logger
from shutil import rmtree
from natsort import natsorted


class DataLoader:
    def __init__(self, config):
        """
        Args:
            data_dir: this folder path should contain both Anomalous and Normal images
        """
        self.config = config
        self.image_size = self.config.data_loader.image_size
        log_object = Logger(self.config)
        self.logger = log_object.get_logger(__name__)
        dataset_type = self.config.data_loader.dataset_name
        if dataset_type == "material":
            self.build_material_dataset()
        elif dataset_type == "cifar10":
            self.build_cifar10_dataset()
        elif dataset_type == "mnist":
            self.build_mnist_dataset()

    def build_material_dataset(self):
        self.data_dir = self.config.dirs.data
        self.train = "train_{}".format(self.image_size)
        self.test = "test_{}".format(self.image_size)
        self.test_vis = "test_vis"
        self.test_vis_big ="test_vis_big"
        self.valid = "valid_{}".format(self.image_size)
        self.train_dataset = os.path.join(self.data_dir, self.train)
        self.valid_dataset = os.path.join(self.data_dir, self.valid)
        self.img_location = os.path.join(self.data_dir, self.test, "imgs/")
        self.img_location_vis = os.path.join(self.data_dir, self.test_vis, "imgs/")
        self.img_location_vis_big = os.path.join(self.data_dir, self.test_vis_big, "imgs/")
        self.tag_location = os.path.join(self.data_dir, self.test, "labels/")
        self.tag_location_vis = os.path.join(self.data_dir, self.test_vis, "labels/")
        self.tag_location_vis_big = os.path.join(self.data_dir, self.test_vis_big, "labels/")
        if not os.path.exists(self.data_dir):
            self.logger.info("Dataset is not present. Download is started.")
            download_data_material(self.data_dir)
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
            self.populate_train_material()
        else:
            self.populate_train_valid_material()
        if self.config.data_loader.mode == "anomaly":
            self.populate_test_material()
        if self.config.data_loader.mode == "visualization":
            self.populate_test_material_vis()
        if self.config.data_loader.mode == "visualization_big":
            self.populate_test_material_vis_big()

    def build_cifar10_dataset(self):
        pass

    def build_mnist_dataset(self):
        pass

    def populate_train_material(self):
        # Check if we have the data already
        if self.train in self.dir_names:
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

    def populate_train_valid_material(self):
        if self.train in self.dir_names and self.valid in self.dir_names:
            self.logger.info("Train and Validation datasets are already populated")
        else:
            # Remove train dataset from the previous run
            if os.path.exists(self.train_dataset):
                rmtree(self.train_dataset)
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

    def populate_test_material(self):
        if self.test in self.dir_names:
            self.logger.info("Test Dataset is already populated")
        else:
            self.logger.info("Test Dataset will be populated")
            size = self.config.data_loader.image_size
            folder_name = self.test
            first_level = os.path.join(self.data_dir, folder_name)
            if not os.path.exists(first_level):
                os.mkdir(first_level)
            img_files = []
            tag_files = []
            for img_, tag_ in self.image_tag_list:
                h, w = img_.shape[:2]
                self.w_turns = (w // size) * 2 - 1
                self.h_turns = (h // size) * 2 - 1
                slide = int(size / 2)
                for adv_h in range(self.h_turns):
                    for adv_w in range(self.w_turns):
                        image = img_[
                            adv_h * slide : size + ((adv_h) * slide),
                            adv_w * slide : ((adv_w) * slide) + size,
                        ]
                        tag = tag_[
                            adv_h * slide : size + ((adv_h) * slide),
                            adv_w * slide : ((adv_w) * slide) + size,
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

    def populate_test_material_vis(self):
        if self.test_vis in self.dir_names:
            self.logger.info("Test Dataset is already populated")
        else:
            self.logger.info("Test Dataset will be populated")
            size = self.config.data_loader.image_size
            folder_name = self.test_vis
            first_level = os.path.join(self.data_dir, folder_name)
            if not os.path.exists(first_level):
                os.mkdir(first_level)
            img_files = []
            tag_files = []
            for img_, tag_ in self.image_tag_list:
                h, w = img_.shape[:2]
                self.w_turns = (w // size)
                self.h_turns = (h // size)
                slide = int(size)
                for adv_h in range(self.h_turns):
                    for adv_w in range(self.w_turns):
                        image = img_[
                            adv_h * slide : size + ((adv_h) * slide),
                            adv_w * slide : ((adv_w) * slide) + size,
                        ]
                        tag = tag_[
                            adv_h * slide : size + ((adv_h) * slide),
                            adv_w * slide : ((adv_w) * slide) + size,
                        ]
                        img_files.append(image)
                        tag_files.append(tag)
            self.test_size_per_img = self.w_turns * self.h_turns
            if not os.path.exists(self.img_location_vis):
                os.mkdir(self.img_location_vis)
            with working_directory(self.img_location_vis):
                for idx, img in enumerate(img_files):
                    im = Image.fromarray(img)
                    im.save(
                        "{}.jpg".format(
                            idx
                        )
                    )
            if not os.path.exists(self.tag_location_vis):
                os.mkdir(self.tag_location_vis)
            with working_directory(self.tag_location_vis):
                for idx, tag in enumerate(tag_files):
                    im = Image.fromarray(tag)
                    im.save(
                        "{}.jpg".format(
                            idx
                        )
                    )
                    
    def populate_test_material_vis_big(self):
        if self.test_vis_big in self.dir_names:
            self.logger.info("Test Dataset is already populated")
        else:
            self.logger.info("Test Dataset will be populated")
            size = self.config.data_loader.image_size
            folder_name = self.test_vis_big
            first_level = os.path.join(self.data_dir, folder_name)
            if not os.path.exists(first_level):
                os.mkdir(first_level)
            img_files = []
            tag_files = []
            #index_list = [0,6,7,8,9,10,11,15]
            index_list = [0,6,7,8,9,10,11,15]
            image_tags = [self.image_tag_list[i] for i in index_list]
            for img_, tag_ in image_tags:
                h, w = img_.shape[:2]
                self.w_turns = w - size + 1
                self.h_turns = h - size + 1
                slide = 1
                for adv_h in range(self.h_turns):
                    for adv_w in range(self.w_turns):
                        image = img_[
                            adv_h * slide : size + ((adv_h) * slide),
                            adv_w * slide : ((adv_w) * slide) + size,
                        ]
                        tag = tag_[
                            adv_h * slide : size + ((adv_h) * slide),
                            adv_w * slide : ((adv_w) * slide) + size,
                        ]
                        img_files.append(image)
                        tag_files.append(tag)
            self.test_size_per_img = self.w_turns * self.h_turns
            if not os.path.exists(self.img_location_vis_big):
                os.mkdir(self.img_location_vis_big)
            with working_directory(self.img_location_vis_big):
                for idx, img in enumerate(img_files):
                    im = Image.fromarray(img)
                    im.save(
                        "{}.jpg".format(
                            idx
                        )
                    )
            if not os.path.exists(self.tag_location_vis_big):
                os.mkdir(self.tag_location_vis_big)
            with working_directory(self.tag_location_vis_big):
                for idx, tag in enumerate(tag_files):
                    im = Image.fromarray(tag)
                    im.save(
                        "{}.jpg".format(
                            idx
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
            labels.append(1) if np.sum(im2arr) > 5100 else labels.append(0)
        labels_f = tf.constant(labels)
        self.logger.info("Test Dataset is Loaded")
        return [img_names, labels_f]

    def get_test_dataset_vis(self):
        """
        :param size: size of the image
        :return: numpy array of images and corresponding labels
        """
        img_list = listdir_nohidden(self.img_location_vis)
        img_list = natsorted(img_list)
        img_names = tf.constant([os.path.join(self.img_location_vis, x) for x in img_list])
        tag_list = listdir_nohidden(self.tag_location_vis)
        tag_list = natsorted(tag_list)
        tag_list_merged = [os.path.join(self.tag_location_vis, x) for x in tag_list]
        labels = []
        for label in tag_list_merged:
            im2arr = io.imread(label)
            labels.append(1) if np.sum(im2arr) > 5100 else labels.append(0)
        labels_f = tf.constant(labels)
        self.logger.info("Test Dataset is Loaded")
        return [img_names, labels_f, tag_list_merged]
    
    def get_test_dataset_vis_big(self):
        """
        :param size: size of the image
        :return: numpy array of images and corresponding labels
        """
        img_list = listdir_nohidden(self.img_location_vis_big)
        img_list = natsorted(img_list)
        #img_list = img_list[660345:660345 * 2]
        img_names = tf.constant([os.path.join(self.img_location_vis_big, x) for x in img_list])
        tag_list = listdir_nohidden(self.tag_location_vis_big)
        tag_list = natsorted(tag_list)
        #tag_list = tag_list[660345:660345* 2]
        tag_list_merged = [os.path.join(self.tag_location_vis_big, x) for x in tag_list]
        labels = []
        for label in tag_list_merged:
            im2arr = io.imread(label)
            labels.append(1) if np.sum(im2arr) > 5100 else labels.append(0)
        labels_f = tf.constant(labels)
        self.logger.info("Test Dataset is Loaded")
        return [img_names, labels_f, tag_list_merged]
