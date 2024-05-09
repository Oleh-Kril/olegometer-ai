import os
import csv
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.image as mpimg

folder1_path = "./real-dataset/segmented-designs/"
folder2_path = "./real-dataset/segmented-pages/"

# List all the files in the folders
folder1_files = os.listdir(folder1_path)
folder2_files = os.listdir(folder2_path)

N_to_skip = 0
folder1_files = folder1_files[N_to_skip:]




for file1_name in folder1_files:
    image1_path = os.path.join(folder1_path, file1_name)
    for file2_name in folder2_files:
        image2_path = os.path.join(folder2_path, file2_name)

        # Display images

