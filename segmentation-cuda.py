import os
import re
import time
import multiprocessing
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
from PIL import Image

import cupy as cp
from skimage import io
from cucim.skimage.color import rgb2gray
from cucim.skimage.morphology import disk
from cucim.skimage.measure import regionprops, label
from cucim.skimage.feature import canny

plt.rcParams["figure.figsize"] = (12, 8)

def edge_segmentation(image_path):
    # load images and convert to grayscale
    image = io.imread(image_path)
    image = cp.array(image)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image_wh = rgb2gray(image)

    edges = canny(image_wh)

    dilated_edges = nd.binary_dilation(edges, disk(8))

    # fill regions to perform edge segmentation
    fill_im = nd.binary_fill_holes(dilated_edges)

    # Label connected components
    labeled_image = label(fill_im, connectivity=2, background=0)

    return labeled_image, image, edges

base = "./drive/MyDrive/"
root_folder = base + "pages-designs-flat"
output_folder = base + "segmented-pages-canny"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

def cut_and_save(image_name, labeled_image, image, edges):
    # Get region properties for each labeled region in both images
    regions = regionprops(labeled_image)

    for i, region in enumerate(regions):
        bbox = region.bbox
        image_ui_element = image[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
        # Save image_ui_element
        output_path = os.path.join(output_folder, f"{image_name}_{i}.jpeg")
        # skimage.io.imsave(output_path, image_ui_element)


def process_image(image_path):
    image_name = re.sub(r'^\.\/', '', image_path.split('/')[-1])
    labeled_image, image, edges = edge_segmentation(image_path)
    cut_and_save(image_name, labeled_image, image, edges)


def traverse_folder(folder_path):
    # Iterate over all items in the directory
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        print(item_path)
        process_image(item_path)


# Main function to start the traversal from a given root folder
def main():
    traverse_folder(root_folder)

# Example usage:
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time}")
