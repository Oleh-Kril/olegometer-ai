import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.measure import regionprops, label
from skimage.transform import resize
from sklearn.metrics.pairwise import cosine_similarity

from edge_segmentation import edge_segmentation, show_groups
from skimage import metrics

plt.rcParams["figure.figsize"] = (12, 8)
original_img_path = '../dataset/5a.png'
img_to_compare_path = '../dataset/5b.png'

def show_two_images(img1, img2):
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title('Image 1')

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title('Image 2')

    plt.show()

def compare_dimensions(region1, region2):
    height1, width1 = region1[2] - region1[0], region1[3] - region1[1]
    height2, width2 = region2[2] - region2[0], region2[3] - region2[1]

    return max(height1, height2), max(width1, width2)

def flatten_and_concatenate(image, bbox, resize_w, resize_h):
    region = image[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
    resized_region = resize(region, (resize_h, resize_w), mode='constant', anti_aliasing=True)

    flattened_channelR = resized_region[:,:,0].ravel()
    flattened_channelG = resized_region[:,:,1].ravel()
    flattened_channelB = resized_region[:,:,2].ravel()

    return np.concatenate([flattened_channelR, flattened_channelG, flattened_channelB])

def detect_and_match_features(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect Harris corners
    corners1 = cv2.cornerHarris(gray1, 2, 3, 0.04)
    corners2 = cv2.cornerHarris(gray2, 2, 3, 0.04)

    # Threshold for corner detection
    threshold = 0.01
    corners1[corners1 > threshold * corners1.max()] = 255
    corners1[corners1 <= threshold * corners1.max()] = 0

    corners2[corners2 > threshold * corners2.max()] = 255
    corners2[corners2 <= threshold * corners2.max()] = 0

    # Find key points using the corners
    keypoints1 = np.argwhere(corners1 == 255)
    keypoints1 = [cv2.KeyPoint(float(x[1]), float(x[0]), 1) for x in keypoints1]

    keypoints2 = np.argwhere(corners2 == 255)
    keypoints2 = [cv2.KeyPoint(float(x[1]), float(x[0]), 1) for x in keypoints2]

    # Create ORB descriptor
    orb = cv2.ORB_create()

    # Compute descriptors and keypoints using ORB
    keypoints1, descriptors1 = orb.compute(gray1, keypoints1)
    keypoints2, descriptors2 = orb.compute(gray2, keypoints2)
    if descriptors1 is not None and descriptors2 is not None:
        # Create BFMatcher (Brute-Force Matcher)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(descriptors1, descriptors2)

        # Sort them in ascending order of distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw matches on images
        img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        if(len(matches) > 0):
            cv2.imshow('Matches', img_matches)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return len(matches) > 0
    else:
        return False

def compare_and_highlight(labeled_original, labeled_to_compare, image_original, image_to_compare, edges_original, edges_to_compare, epsilon=0.2):
    to_compare_mask_with_highlighted_regions = np.zeros_like(labeled_to_compare)
    region_dict = {}

    # Get region properties for each labeled region in both images
    regions_of_original = regionprops(labeled_original)
    regions_of_to_compare = regionprops(labeled_to_compare)

    # regions_of_original = sorted(regions_of_original, key=lambda x: (x.area),reverse=True)
    # regions_of_to_compare = sorted(regions_of_to_compare,key=lambda x: (x.area), reverse=True)

    i = 1  # Indices for iterating through regions_of_original and regions_of_to_compare

    for region_of_original in regions_of_original:
        bbox_of_original = region_of_original.bbox

        # Iterate over each element in regions_of_to_compare
        for region_of_to_compare in regions_of_to_compare:
            bbox_of_to_compare = region_of_to_compare.bbox

            # resize_h, resize_w = compare_dimensions(bbox_of_original, bbox_of_to_compare)

            image_group_original = image_original[bbox_of_original[0]:bbox_of_original[2], bbox_of_original[1]:bbox_of_original[3], :]
            image_group_to_compare = image_to_compare[bbox_of_to_compare[0]:bbox_of_to_compare[2], bbox_of_to_compare[1]:bbox_of_to_compare[3], :]

            if detect_and_match_features(image_group_original, image_group_to_compare):
                minr, minc, maxr, maxc = bbox_of_to_compare
                to_compare_mask_with_highlighted_regions[minr:maxr, minc:maxc] = labeled_to_compare[minr:maxr,minc:maxc]

                original_element = image_original[bbox_of_original[0]:bbox_of_original[2], bbox_of_original[1]:bbox_of_original[3], :]
                to_compare_element = image_to_compare[bbox_of_to_compare[0]:bbox_of_to_compare[2], bbox_of_to_compare[1]:bbox_of_to_compare[3], :]
                show_two_images(image_group_original, image_group_to_compare)
                regions_of_to_compare.remove(region_of_to_compare)

                # Print the pair of elements
                print("Original Bbox:", bbox_of_original)
                print("To Compare Bbox:", bbox_of_to_compare)
                # print(i, "\n")
                i += 1
                break


    # while i < len(regions_of_original) and j < len(regions_of_to_compare):
    #     bbox_of_original = regions_of_original[i].bbox
    #     bbox_of_to_compare = regions_of_to_compare[j].bbox
    #
    #     size_diff = abs(regions_of_to_compare[j].area_bbox - regions_of_original[i].area_bbox)
    #     position_diff = sum(
    #         [
    #             abs(bbox_of_original[0] - bbox_of_to_compare[0]),
    #             abs(bbox_of_original[1] - bbox_of_to_compare[1]),
    #             abs(bbox_of_original[2] - bbox_of_to_compare[2]),
    #             abs(bbox_of_original[3] - bbox_of_to_compare[3]),
    #         ]
    #     )
    #
    #
    #     # Check if the regions are similar based on coordinates and sizes with epsilon tolerance
    #     if size_diff >= regions_of_original[i].area_bbox*epsilon:
    #         region_dict[j] = f"size difference is {size_diff}"
    #
    #         # Highlight the region on image to compare
    #         minr, minc, maxr, maxc = bbox_of_original
    #         to_compare_mask_with_highlighted_regions[minr:maxr, minc:maxc] = labeled_to_compare[minr:maxr, minc:maxc]
    #         i += 1
    #     elif position_diff >= epsilon*1000:
    #         region_dict[j] = f"position_diff difference is {position_diff}"
    #
    #         # Highlight the region in image2 that is absent in image1
    #         minr, minc, maxr, maxc = bbox_of_original
    #         to_compare_mask_with_highlighted_regions[minr:maxr, minc:maxc] = labeled_to_compare[minr:maxr, minc:maxc]
    #         i += 1
    #     else:
    #         i += 1
    #         j += 1

    # Highlight any remaining regions in image2
    # while j < len(regions_of_to_compare):
    #     bbox_of_to_compare = regions_of_to_compare[j].bbox
    #     minr, minc, maxr, maxc = bbox_of_to_compare
    #     to_compare_mask_with_highlighted_regions[minr:maxr, minc:maxc] = labeled_to_compare[minr:maxr, minc:maxc]
    #     j += 1

    # print(region_dict)
    return to_compare_mask_with_highlighted_regions, region_dict


labeled_original, img_original, edges_original = edge_segmentation(image_path=original_img_path)
labeled_to_compare, img_to_compare, edges_to_compare = edge_segmentation(image_path=img_to_compare_path)

ui_elements_on_original = np.max(labeled_original)
ui_elements_on_to_compare = np.max(labeled_to_compare)

if ui_elements_on_original != ui_elements_on_to_compare - 100:
    print(f"Images aren't the same because expected: {ui_elements_on_original} ui elements, but got {ui_elements_on_to_compare}")

    to_compare_mask_with_highlighted_regions, regions_difference_info = compare_and_highlight(
        labeled_original, labeled_to_compare, img_original, img_to_compare, edges_original, edges_to_compare
    )
    show_groups(img_to_compare, to_compare_mask_with_highlighted_regions)
else:
    print(f"Images are the same. Number of groups {ui_elements_on_original}")

# show_groups(img_original, labeled_original)
# show_groups(img_to_compare, labeled_to_compare)