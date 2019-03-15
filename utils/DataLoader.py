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


def show_image_from_path(image):
    """
    Args:
        image: absolute path of the image
    """
    plt.imshow(io.imread(image), cmap="gray")


def show_image_from_memory(image):
    """
    Args:
        image: io.imread object
    """
    plt.imshow(image, cmap="gray")


class DataLoader():

    def __init__(self, data_dir):
        """
        Args:
            data_dir: this folder path should contain both Anomalous and Normal images
        """
        self.data_dir = data_dir
        self.dataset_name= None
        # this is to list all the folders
        self.dir_names =listdir_nohidden(self.data_dir)
        # If the cropped subsets are not present create them
        self.dataset_list = []
        # Download the data if it is not downloaded
        if not os.path.exists(self.data_dir):
            print("DataLoader: dataset is not present. Download is started.")
            download_data(self.data_dir)
        normal_imgs = self.data_dir + "/Normal/"
        anorm_imgs = self.data_dir + "/Anomalous/images/"
        norm_img_nms = [normal_imgs + x for x in listdir_nohidden(normal_imgs)]
        anorm_img_nms = [anorm_imgs + x for x in listdir_nohidden(anorm_imgs)]
        self.norm_img_array = self.create_image_array(norm_img_nms,save=False)
        self.anorm_img_array = self.create_image_array(anorm_img_nms,save=False)
        self.populate()
    
    def populate(self):
        if len(self.dir_names) == 2:
            print("DataLoader: Cropped subsets will be populated")
            size_list = [28]
            folder_name = "cropped"
            num_images = 5000
            for size in size_list:
                folder = folder_name + str(size)
                self.dataset_list.append(folder)
                self.generate_sub_dataset(self.norm_img_array, size=size, num_images=num_images, save=True,
                                          folder_name=folder)
        else:
            print("DataLoader: Subsets are already populated.")

    def create_image_array(self, img_names,save=True,file_name="Dataset",size=28,progress=False):
        """
        Args:
            img_names:
        """
        self.dataset_name = self.data_dir + "/" + file_name + "-" + str(size)
        img_array = []
        if progress:
            for img in tqdm(img_names):
                im2arr = io.imread(img)
                img_array.append(im2arr)
        else:
            for img in img_names:
                im2arr = io.imread(img)
                img_array.append(im2arr)
        if save:
            np.save(self.dataset_name,img_array)
        return np.array(img_array)

    def generate_sub_dataset(self, image_array, size, num_images, save=False, folder_name="cropped"):
        """
        Args:
            image_array: image array of the dataset
            size: size of the image
            num_images: total number of images
            save: whether to save the generated subset or not
            folder_name: output folder name
        """
        print("DataLoader: generating new dataset with size:{}".format(size))
        imgs = []
        for ind, img in enumerate(image_array):
            h, w = img.shape[:2]
            new_h, new_w = size, size
            for idx in range(num_images):
                top = np.random.randint(0, h - new_h)
                left = np.random.randint(0, w - new_w)
                image = img[top: top + new_h, left:left + new_w]
                imgs.append(image)
            print("{} images generated".format(num_images * (ind + 1)))
        if save:
            with working_directory("./data"):
                if not os.path.exists(folder_name):
                    os.mkdir(folder_name)
                    with working_directory(folder_name):
                        for idx, img in enumerate(imgs):
                            im = Image.fromarray(img)
                            im.save("img-" + str(idx) + ".jpg")
                else:
                    print("{} exists".format(folder_name))


        return imgs

    def get_sub_dataset(self, size):
        """

        :param size: size of the image
        :return: numpy array of images and corresponding labels
        """
        folder_name = self.data_dir + "/cropped" + str(size) + "/"
        img_names = []
        labels = []
        if os.path.exists(folder_name):
            img_list = listdir_nohidden(folder_name)
            img_names = tf.constant([folder_name + x for x in img_list])
            labels = tf.constant([x[4:-4] for x in img_list])
        else:
            print("DataLoader: subset doesn't exist")
        return [img_names, labels]
