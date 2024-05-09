from skimage import metrics
from skimage import io, color
from skimage.transform import resize
import numpy as np

import cv2
def histogram_intersection(hist1, hist2):
    """
    Compute histogram intersection between two histograms.
    """
    return np.sum(np.minimum(hist1, hist2))


def calculate_histogram(image):
    """
    Calculate the histogram of an image for each color channel.
    """
    histograms = []

    for channel in range(image.shape[2]):
        # Calculate histogram for each color channel
        hist, _ = np.histogram(image[:, :, channel].flatten(), bins=256, range=[0, 256])

        # Normalize histogram
        hist = hist / np.sum(hist)

        histograms.append(hist)

    return histograms
# Load images
image1 = io.imread('1-diff.png')
image2 = io.imread('2-diff.png')

# Keep only the RGB channels
# image1 = image1[:, :, :3]
# image2 = image2[:, :, :3]

first_image_hist = cv2.calcHist([image1], [0], None, [256], [0, 256])
second_image_hist = cv2.calcHist([image2], [0], None, [256], [0, 256])

img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
img_template_probability_match = cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
img_template_diff = 1 - img_template_probability_match

# taking only 10% of histogram diff, since it's less accurate than template method
commutative_image_diff = (img_hist_diff / 10) + img_template_diff
print(commutative_image_diff)