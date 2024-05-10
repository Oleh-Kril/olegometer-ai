import os
import re
import skimage.io
from skimage.measure import regionprops
from edge_segmentation import edge_segmentation, show_groups

root_folder = "./real-dataset/pages-original"
output_folder = "./real-dataset/segmented-pages-1600"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

def cut_and_save(image_name, labeled_image, image, edges):
    # Get region properties for each labeled region in both images
    regions = regionprops(labeled_image)

    for i, region in enumerate(regions):
        bbox = region.bbox
        image_ui_element = image[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
        # Save image_ui_element
        output_path = os.path.join(output_folder, f"{i}.jpeg")
        skimage.io.imsave(output_path, image_ui_element)


def process_image(image_path):
    image_name = re.sub(r'^\.\/', '', image_path.split('/')[-1])
    labeled_image, image, edges = edge_segmentation(image_path)
    cut_and_save(image_name, labeled_image, image, edges)


def traverse_folder(folder_path):
    # Iterate over all items in the directory
    for item in os.listdir(folder_path):
        # Get full path of the item
        item_path = os.path.join(folder_path, item)

        print(item_path)

        # If it's a file, process it
        if os.path.isfile(item_path) and item.startswith("1600"):
            process_image(item_path)
        # If it's a directory, recursively traverse it
        elif os.path.isdir(item_path):
            traverse_folder(item_path)


# Main function to start the traversal from a given root folder
def main(root_folder):
    traverse_folder(root_folder)


# Example usage:
if __name__ == "__main__":
    main(root_folder)
