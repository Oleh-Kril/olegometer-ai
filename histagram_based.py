import numpy as np

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

        histograms.append(hist)

    return histograms

def get_similarity(image1, image2):
    histograms1 = calculate_histogram(image1)
    histograms2 = calculate_histogram(image2)

    # Calculate histogram intersection for each channel
    intersection = np.sum([histogram_intersection(hist1, hist2) for hist1, hist2 in zip(histograms1, histograms2)])

    return intersection