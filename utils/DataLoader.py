
import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
from utils.utils import working_directory

class DataLoader(object):

    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.image_names = os.listdir(self.data_dir)
        self.merged = [self.data_dir + x for x in self.image_names]
        self.images_orig = self.create_image_array(self.merged)
        self.dataset_list = []

    def show_image_from_memory(self,image):
        plt.imshow(image, cmap="gray")
    
    def show_image_from_path(self,image):
        plt.imshow(io.imread(image),cmap="gray")

    def create_image_array(self,img_names):
        img_array = []
        for img in img_names:
            im2arr = io.imread(img)
            img_array.append(im2arr)
        return np.array(img_array)

    def generate_sub_dataset(self,size,num_images,save=False,folder_name="cropped"):
        print("DataLoader: generating new dataset")
        imgs = []
        for ind,img in enumerate(self.images_orig):
            h, w = img.shape[:2]
            new_h, new_w = size, size
            for idx in range(num_images):
                top = np.random.randint(0, h - new_h)
                left = np.random.randint(0, w - new_w)
                image = img[top: top + new_h, left:left + new_w]
                imgs.append(image)
            print("{} images generated".format(num_images * (ind+1)))
        if save:
            with working_directory("./data"):
                # TODO check if it is created
                os.mkdir(folder_name)
            with working_directory("./data/" + folder_name):
                for idx,img in enumerate(imgs):
                    im = Image.fromarray(img)
                    im.save("img-" +str(idx)+ ".tif")
        return imgs

    def build_image_repo(self,size_list,num_images,folder_name):
        for size in size_list:
            folder = folder_name+str(size)
            self.dataset_list.append(folder)
            self.generate_sub_dataset(size=size,num_images=num_images,save=True,folder_name=folder)

    def get_original_images(self):
        return self.images_orig

    def get_sub_datasets(self):
        for folder in self.dataset_list:
            print(folder)
        return self.dataset_list

