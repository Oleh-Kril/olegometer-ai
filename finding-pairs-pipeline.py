import csv
import os
from PIL import Image
from matplotlib import pyplot as plt, image as mpimg
from matplotlib.widgets import Button
from skimage.transform import resize
from ssim_similarity import get_similarity
from skimage import io, color

def save_and_next(event):
    global original_path, found_path, labels_file

    filename1 = os.path.basename(original_path)
    filename2 = os.path.basename(found_path)
    # Write to labels.csv
    with open(labels_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([filename1, filename2])

    plt.close()  # Close the current plot window

def next_image(event):
    plt.close()  # Close the current plot window

def resize_and_check_ratio(image1, image2):
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    if width1 > width2:
        ratio = width1 / width2
    else:
        ratio = width2 / width1

    if ratio > 2:
        return None, None

    if height1 > height2:
        ratio = max(ratio, height1 / height2)
    else:
        ratio = max(ratio, height2 / height1)

    if ratio > 2:
        return None, None

    # Resize the images
    max_height = max(height1, height2, 100)
    max_width = max(width1, width2, 100)

    image1 = resize(image1, (max_height, max_width), mode='constant', anti_aliasing=True)
    image2 = resize(image2, (max_height, max_width), mode='constant', anti_aliasing=True)

    return image1, image2

# Define the paths to the folders
folder1_path = "./real-dataset/segmented-designs/"
folder2_path = "./real-dataset/segmented-pages/"
labels_file = "labels_cosine.csv"

# pairs_dict = {}
# if os.path.exists(labels_file):
#     with open(labels_file, 'r') as csvfile:
#         reader = csv.reader(csvfile)
#         for row in reader:
#             pairs_dict[row[0]] = row[1]

# List all the files in the folders
folder1_files = os.listdir(folder1_path)
folder2_files = os.listdir(folder2_path)

min_accessible_similarity = 0.5

N_to_skip = 22

folder1_files = folder1_files[N_to_skip:]

correct = 0
all = 0
for file1_name in folder1_files:
    all += 1
    j = 0
    max_similarity = 0
    max_similarity_pair = []

    for file2_name in folder2_files:
        if j % 20 == 0:
            print(f"Current iteration: {j}")
        j += 1

        image1_path = os.path.join(folder1_path, file1_name)
        image2_path = os.path.join(folder2_path, file2_name)

        image1 = io.imread(image1_path)
        image2 = io.imread(image2_path)

        image1, image2 = resize_and_check_ratio(image1, image2)
        if image1 is None or image2 is None:
            continue

        # Call the function to get similarity
        similarity_score = get_similarity(image1, image2)

        if similarity_score > max_similarity:
            max_similarity = similarity_score
            # print(f"Change of pair on iteration {j}. Max similarity now is {max_similarity}")
            # print("Current pair images")
            max_similarity_pair = [(image1_path, image2_path)]
            # print(max_similarity_pair[-1])


    if max_similarity > min_accessible_similarity:
        original_path, found_path = max_similarity_pair[-1]
        # filename1 = os.path.basename(original_path)
        # filename2 = os.path.basename(found_path)

        img1 = mpimg.imread(image1_path)
        img2 = mpimg.imread(image2_path)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img1)
        ax[0].axis('off')
        ax[0].set_title('Image 1')
        ax[1].imshow(img2)
        ax[1].axis('off')
        ax[1].set_title('Image 2')

        # Create buttons
        axsave = plt.axes([0.1, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.3, 0.05, 0.1, 0.075])
        bsave = Button(axsave, 'Save')
        bnext = Button(axnext, 'Next')
        bsave.on_clicked(save_and_next)
        bnext.on_clicked(next_image)

        plt.show()
        # if filename1 in pairs_dict:
        #     if pairs_dict[filename1] == filename2:
        #         correct += 1
        #         print(f"Correct for sure: {correct}/{all}")
