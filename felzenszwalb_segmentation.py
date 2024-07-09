from scipy.ndimage import label
from skimage import io, color, segmentation

# Load image from filesystem
def felzenszwalb_segmentation(image_path):
    image = io.imread(image_path)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    # Convert to grayscale
    image_gray = color.rgb2gray(image)

    # Apply Felzenszwalb segmentation
    segments_felzenszwalb = segmentation.felzenszwalb(image, scale=100, sigma=0.5, min_size=50)

    labeled_segments, num_groups = label(segments_felzenszwalb)

    return labeled_segments, num_groups
