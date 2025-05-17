import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import canny
from skimage import io
from skimage.color import rgb2gray
import scipy.ndimage as nd
from skimage.measure import regionprops, label
from skimage.morphology import disk

plt.rcParams["figure.figsize"] = (12, 8)

def edge_segmentation(image_path = '../images/9b.png'):
    # load images and convert to grayscale
    image = cv2.imread(image_path)

    if image.shape[-1] == 4:
        image = image[:, :, :3]
    plt.imshow(image)
    plt.title('image without alpha channel')
    # plt.show()

    image_wh = rgb2gray(image)

    # apply edge segmentation
    # plot canny edge detection
    edges = canny(image_wh)
    plt.imshow(edges, interpolation='gaussian')
    plt.title('Canny detector')
    # plt.show()

    dilated_edges = nd.binary_dilation(edges, disk(8))

    plt.imshow(dilated_edges, interpolation='gaussian')
    plt.title('Combined edges')
    # plt.show()

    # fill regions to perform edge segmentation
    fill_im = nd.binary_fill_holes(dilated_edges)
    plt.imshow(fill_im)
    plt.title('Region Filling')
    # plt.show()

    # Label connected components
    labeled_image = label(fill_im, connectivity=2, background=0)

    return labeled_image, image, edges

def show_groups(image, labeled_image):
    # Create a color map for visualization
    num_labels = np.max(labeled_image)
    colors = plt.cm.jet(np.linspace(0, 1, num_labels + 1))

    # Visualize the labeled regions in different colors
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)

    for region in regionprops(labeled_image):
        # Draw rectangle around segmented region
        minr, minc, maxr, maxc = region.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                             fill=False, edgecolor=colors[region.label], linewidth=2)
        ax.add_patch(rect)

    plt.title('Region Filling with Color-Coded Groups')
    plt.show()

if __name__ == '__main__':
    labeled_image, image = edge_segmentation()
    show_groups(image, labeled_image)
