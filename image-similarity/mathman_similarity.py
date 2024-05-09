from skimage import metrics
from skimage import io, color
from skimage.transform import resize

def map_range(value, input_min, input_max, output_min, output_max):
    # Map the value from the input range to the output range
    return ((value - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min


# Load images
image1 = io.imread('a.jpeg')
image2 = io.imread('b-color-diff.jpeg')

# Keep only the RGB channels
image1 = image1[:, :, :3]
image2 = image2[:, :, :3]

# Resize images
image1 = resize(image1, (200, 200), mode='constant', anti_aliasing=True)
image2 = resize(image2, (200, 200), mode='constant', anti_aliasing=True)

# Convert images to grayscale
gray_image1 = color.rgb2gray(image1)
gray_image2 = color.rgb2gray(image2)

# Calculate Structural Similarity Index with data_range parameter
ssi_index, _ = metrics.structural_similarity(image1, image2, channel_axis=2, full=True, data_range=gray_image1.max() - gray_image1.min())
# ssi_index, _ = metrics.structural_similarity(gray_image1, gray_image2, full=True, data_range=gray_image1.max() - gray_image1.min())

# Convert similarity index to a distance-like measure
ssi_similarity = map_range(ssi_index, -1, 1, 0, 1)

print(f"Structural Similarity: {ssi_similarity}")
