import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, segmentation, morphology, measure
from skimage.color import label2rgb
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

plt.rcParams["figure.figsize"] = (12, 8)

# Load image from filesystem
image_path = '../dataset/images/4a.png'
image = io.imread(image_path)

if image.shape[-1] == 4:
    image = image[:, :, :3]
# Convert to grayscale

image_gray = color.rgb2gray(image)

# Text detection using simple thresholding (you may need to fine-tune the threshold)
text_mask = image_gray < 0.5

# Edge detection using Canny
edges = filters.sobel(image_gray)

# Combine text mask with edges
combined_mask = np.logical_or(edges, text_mask)

# Apply morphological operations to enhance the mask
combined_mask = morphology.binary_dilation(combined_mask)

# Label connected components in the mask
labeled_mask, num_labels = measure.label(combined_mask, connectivity=2, return_num=True)

# Apply the mask to the original image and display each group in a different color
segmented_image = label2rgb(labeled_mask, image=image, bg_label=0)

# Plot the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

ax1.imshow(image)
ax1.set_title('Original Image')

ax2.imshow(text_mask, cmap='gray')
ax2.set_title('Text Mask')

ax3.imshow(segmented_image)
ax3.set_title('Combined Segmentation')

plt.show()