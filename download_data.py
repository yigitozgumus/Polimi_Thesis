#!/usr/bin python
from urllib import request
from zipfile import ZipFile
from utils.utils import working_directory
import os

def download_data():
    print("Beginning data download...")
    dir_path = "./data"
    dir_name = "data.zip"
    if os.path.exists(dir_path):
        print("Data is already downloaded")
    else:
        os.mkdir(dir_path)
        url ="http://web.mi.imati.cnr.it/ettore/NanoTWICE/dfg05er_Data.zip"
        with working_directory(dir_path):
            request.urlretrieve(url, dir_name)
            with ZipFile(dir_name, 'r') as zip:
                # printing all the contents of the zip file
                #zip.printdir()
                print('Extracting all the files now...')
                zip.extractall()
            os.remove(dir_name)
        print("Data Download completed.")

if __name__ == "__main__":
    download_data()
