import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
from visual_test.custom_types import DiffType
from visual_test.image_utils import phash_region, contains
from visual_test.masking import create_rectangular_diff_mask
from visual_test.element_detection import detect_elements_multiscale
from visual_test.element_matching import match_elements
# from visual_test.pixel_based_matching import match_elements
from visual_test.comparison import compare_positions, compare_sizes, compare_text, filter_nested_missing_elements

# ─── Logging ──────────────────────────────────────────────────────────────────
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger('visual_test')
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

def process_images(img_D: np.ndarray, img_R: np.ndarray, constants: dict | None = None):
    if constants is None:
        constants = {
            'EDGE_DILATION_SIZE': 6,
            'MORPHOLOGY_KERNEL_SIZE': 2,
            'MAX_HASH_MATCHES': 50,
            'HISTOGRAM_WEIGHT': .15,
            'SSIM_WEIGHT': .85,
            'MATCH_THRESHOLD': .9,
            'POSITION_THRESHOLD': 5,
            'SIZE_THRESHOLD': 5,
            'CANNY_LOW_THRESHOLD': 5,
            'CANNY_HIGH_THRESHOLD': 20,
            'MIN_BBOX_AREA': 100,
            'EDGE_DILATION_SIZES':[5],
            'LOCATION_WEIGHT' : 0.15,
            'LOCATION_SIGMA_FRAC' : 0.10,
            'MASK_MIN_DIFF': 99,
        }

    gray_D = cv2.cvtColor(img_D, cv2.COLOR_BGR2GRAY)
    gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)

    # Create and apply masks
    mask = create_rectangular_diff_mask(gray_D, gray_R, constants['MORPHOLOGY_KERNEL_SIZE'])

    # Calculate mask coverage percentage
    mask_coverage = (np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])) * 100
    if mask_coverage > constants['MASK_MIN_DIFF']:
        logger.info(f"Mask coverage {mask_coverage:.2f}% exceeds threshold {constants['MASK_MIN_DIFF']}%, returning no differences")
        return [], []

    elems_D, cont_D, atom_D, edges_D, _ = detect_elements_multiscale(gray_D, constants, mask=mask)
    elems_R, cont_R, atom_R, edges_R, _ = detect_elements_multiscale(gray_R, constants, mask=mask)

    hashes_D = [phash_region(gray_D, e['bbox']) for e in elems_D]
    hashes_R = [phash_region(gray_R, e['bbox']) for e in elems_R]

    matches_DR = match_elements(elems_D, elems_R, gray_D, gray_R, hashes_D, hashes_R, constants, gray_D.shape) #TODO: here shape can be different for gray_R
    # matches_DR = match_elements(elems_D, elems_R, gray_D, gray_R, constants, gray_D.shape) #TODO: here shape can be different for gray_R

    matched_R = set(matches_DR.values())
    diffs_D, diffs_R = [], []
    wrong_pos_seen = set()

    for idD, eD in enumerate(elems_D):
        if idD not in matches_DR:
            diffs_D.append({'type': DiffType.MISSING_ELEMENT, 'bbox': eD['bbox']})
            continue

        idR = matches_DR[idD]
        eR = elems_R[idR]
        parent = next((c['bbox'] for c in cont_D if contains(c['bbox'], eD['bbox'])), None)

        pos_diff = compare_positions(eD, eR, parent, gray_D, gray_R, constants, wrong_pos_seen)
        if pos_diff:
            diffs_R.append(pos_diff)
            continue

        size_diff = compare_sizes(eD, eR, constants)
        if size_diff:
            diffs_R.append(size_diff)

        if eD in atom_D:
            text_diff = compare_text(eD, eR, img_D, img_R)
            if text_diff:
                diffs_R.append(text_diff)

    for j, eR in enumerate(elems_R):
        if j not in matched_R:
            diffs_R.append({'type': DiffType.EXTRA_ELEMENT, 'bbox': eR['bbox']})

    diffs_D = filter_nested_missing_elements(diffs_D)

    # visualize_results(img_D, img_R, diffs_D, diffs_R)

    logger.info(f"Missing on R: {len(diffs_D)}, diffs on R: {len(diffs_R)}")
    return diffs_D, diffs_R

def visualize_results(img_D, img_R, diffs_D, diffs_R):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(cv2.cvtColor(img_D, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Design')
    ax[1].imshow(cv2.cvtColor(img_R, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Real')

    for d in diffs_D:
        x, y, w, h = d['bbox']
        ax[0].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='r', lw=2))
    for d in diffs_R:
        x, y, w, h = d['bbox']
        ax[1].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='r', lw=2))
    plt.show()

def convert_to_simple(dD, dR):
    return {
        'D': {'bboxes': [d['bbox'] for d in dD], 'labels': [d['type'] for d in dD]},
        'R': {'bboxes': [d['bbox'] for d in dR], 'labels': [d['type'] for d in dR]}
    }

if __name__ == '__main__':
    try:
        imgD = cv2.imread('../dataset/images/Case 1.1 D.png')
        imgR = cv2.imread('../dataset/images/Case 1.1 R.png')
        dD, dR = process_images(imgD, imgR)
        print(convert_to_simple(dD, dR))
    except Exception as e:
        logger.error(e) 