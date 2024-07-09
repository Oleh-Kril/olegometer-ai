import os
import re
import time

import skimage.io
from skimage.measure import regionprops
from edge_segmentation import edge_segmentation, show_groups

root_folder_1 = "./real-dataset/designs-original"
root_folder_2 = "./real-dataset/pages-original"
root_folder_flat = "./real-dataset/pages-designs-flat"
output_folder = "./real-dataset/segmented-pages-canny"

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
    labeled_image, image, edges = edge_segmentation(image_path)
    show_groups(image, labeled_image)
    # cut_and_save(image_name, labeled_image, image, edges)


def traverse_folder(folder_path):
    # Iterate over all items in the directory
    for item in os.listdir(folder_path):
        # Get full path of the item
        item_path = os.path.join(folder_path, item)
        print(item_path)

        # If it's a file, process it
        if os.path.isfile(item_path):
            process_image(item_path)
        # If it's a directory, recursively traverse it
        elif os.path.isdir(item_path):
            traverse_folder(item_path)


# Main function to start the traversal from a given root folder
def main():
    # traverse_folder(root_folder_1)
    # traverse_folder(root_folder_2)
    traverse_folder(root_folder_flat)


# Example usage:
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time}")
