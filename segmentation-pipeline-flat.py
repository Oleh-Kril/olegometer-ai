import csv
import os
import re
import time
import multiprocessing

import numpy as np
import skimage.io
from skimage.measure import regionprops
from edge_segmentation import edge_segmentation, show_groups
from felzenszwalb_segmentation import felzenszwalb_segmentation
from color_based_segmentation import histogram_segmentation
root_folder = "./real-dataset/pages-designs-flat"
output_folder = "./real-dataset/segmented-pages-canny"

data_dict = {}

# Open the CSV file
with open('segments.csv', mode='r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)

    # Iterate over the rows in the CSV file
    for row in csv_reader:
        # Use the first column as the key and the second column as the value
        key = row[0]
        value = float(row[1])  # Convert the value to a float (or int if appropriate)
        data_dict[key] = value

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
        skimage.io.imsave(output_path, image_ui_element)


def process_image(image_path):
    image_name = re.sub(r'^\.\/', '', image_path.split('/')[-1])
    # labeled_image, image, edges = edge_segmentation(image_path)
    labeled_image, image, num_labels = histogram_segmentation(image_path)
    labeled_image, num_labels = felzenszwalb_segmentation(image_path)
    # cut_and_save(image_name, labeled_image, image, edges)
    # num_labels = np.max(labeled_image)
    expected_labels = data_dict[image_name]
    return np.abs(num_labels - expected_labels)


def traverse_folder(folder_path, files):
    absolute_differences = []
    for file in files:
        abs_diff = process_image(os.path.join(folder_path, file))
        print(abs_diff)
        absolute_differences.append(abs_diff)
    return np.mean(absolute_differences)


# Main function to start the traversal from a given root folder
def main():
    print("MAE: ", traverse_folder(root_folder, os.listdir(root_folder)))
    # files = os.listdir(root_folder)
    # num_files = len(files)
    # num_processes = min(multiprocessing.cpu_count(), 6)  # Maximum 8 processes
    #
    # chunk_size = (num_files + num_processes - 1) // num_processes
    # chunks = [files[i:i + chunk_size] for i in range(0, num_files, chunk_size)]
    #
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     pool.starmap(traverse_folder, [(root_folder, chunk) for chunk in chunks])


# Example usage:
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time}")
