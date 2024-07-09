import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, segmentation, filters, morphology
from scipy.ndimage import label, binary_dilation, binary_erosion
plt.rcParams["figure.figsize"] = (12, 8)

# Load image from filesystem
# image_path = '../dataset/7b.png'
def histogram_segmentation(image_path):
    image = io.imread(image_path)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    # Convert to HSV color space for better color representation
    image_hsv = color.rgb2hsv(image)

    # Define the color range for segmentation (you may need to adjust these values)
    lower_color = np.array([0.14, 0.14, 0.14])  # Lower bound for color range (HSV)
    upper_color = np.array([1.0, 1.0, 1.0])  # Upper bound for color range (HSV)

    # Create a binary mask based on the color range
    color_mask = np.all((image_hsv >= lower_color) & (image_hsv <= upper_color), axis=-1)

    # Apply morphological closing to enhance the mask (dilation followed by erosion)
    structuring_element = np.ones((5, 5))
    closed_mask = binary_erosion(binary_dilation(color_mask, structure=structuring_element), structure=structuring_element)

    # Optionally apply additional filtering
    closed_mask = filters.rank.mean(closed_mask.astype(float), morphology.disk(20))

    # Apply the mask to the original image
    segmented_image = np.copy(image)
    segmented_image[closed_mask < 0.5] = [255, 255, 255]  # Set non-segmented regions to white

    labeled_image, num = label(closed_mask)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharex=True, sharey=True)
    #
    # ax1.imshow(image)
    # ax1.set_title('Original Image')
    #
    # ax2.imshow(segmented_image)
    # ax2.set_title('Color-Based Segmentation')
    #
    # plt.show()

    return labeled_image, image, num
    # Plot the results
