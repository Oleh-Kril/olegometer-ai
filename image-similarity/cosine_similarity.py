from sklearn.metrics.pairwise import cosine_similarity
from skimage.transform import resize
import numpy as np
from skimage import io

image1 = io.imread('a.jpeg')
image2 = io.imread('b-color-diff.jpeg')

# Keep only the RGB channels
image1 = image1[:, :, :3]
image2 = image2[:, :, :3]

# Resize images
image1 = resize(image1, (200, 200), mode='constant', anti_aliasing=True)
image2 = resize(image2, (200, 200), mode='constant', anti_aliasing=True)

image1_flat = image1.flatten().reshape(1, -1)
image2_flat = image2.flatten().reshape(1, -1)

# Calculate cosine similarity between two regions
similarity = cosine_similarity(image1_flat, image2_flat)[0, 0]

print(similarity)