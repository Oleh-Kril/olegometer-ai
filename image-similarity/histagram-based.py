from skimage import metrics
from skimage import io, color
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

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
        hist, _ = np.histogram(image[:, :, channel].flatten(), bins=256, range=[0, 1])

        # Normalize histogram
        hist = hist / np.sum(hist)

        plt.figure()
        plt.plot(hist, color='black')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.title('Histogram')
        plt.grid(True)
        plt.show()

        histograms.append(hist)

    return histograms
# Load images
image1 = io.imread('a.jpeg')
image2 = io.imread('b-shape-diff.jpeg')

# Keep only the RGB channels
image1 = image1[:, :, :3]
image2 = image2[:, :, :3]

# Resize images
image1 = resize(image1, (200, 200), mode='constant', anti_aliasing=True)
image2 = resize(image2, (200, 200), mode='constant', anti_aliasing=True)

# Convert images to grayscale
# gray_image1 = color.rgb2gray(image1)
# gray_image2 = color.rgb2gray(image2)

histograms1 = calculate_histogram(image1)
histograms2 = calculate_histogram(image2)

# Calculate histogram intersection for each channel
intersection = np.sum([histogram_intersection(hist1, hist2) for hist1, hist2 in zip(histograms1, histograms2)])

# Display results
print(f"Histogram Intersection: {intersection}")