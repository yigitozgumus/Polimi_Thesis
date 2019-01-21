from utils.DataLoader import DataLoader

def main():
    data_dir = "data/Normal/"
    size_list = [28]
    d = DataLoader(data_dir)
    d.build_image_repo(size_list=size_list,num_images=10000,folder_name="cropped")


if __name__ == "__main__":
    main()