import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage import io, color, segmentation

plt.rcParams["figure.figsize"] = (12, 8)

# Load image from filesystem
image_path = '../dataset/4b.png'
image = io.imread(image_path)

if image.shape[-1] == 4:
    image = image[:, :, :3]

# Convert to grayscale
image_gray = color.rgb2gray(image)

# Apply Felzenszwalb segmentation
segments_felzenszwalb = segmentation.felzenszwalb(image, scale=100, sigma=0.5, min_size=50)

labeled_segments, num_groups = label(segments_felzenszwalb)

# Plot the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8), sharex=True, sharey=True)

ax1.imshow(image)
ax1.set_title('Original Image')

ax2.imshow(segments_felzenszwalb, cmap='nipy_spectral')
ax2.set_title('Felzenszwalb Segmentation')

ax3.imshow(labeled_segments, cmap='nipy_spectral')
ax3.set_title(f'Labeled Segments (Total: {num_groups})')

plt.show()

print("Number of groups found:", num_groups)
