import os
from os import listdir
from os.path import isfile, join

data_dir = '/Users/mackim/datasets/garment_images/all_images'

for file in listdir(data_dir):
    if isfile(join(data_dir, file)):
        os.rename(join(data_dir, file), join(data_dir, file.replace(' ', '-')))

