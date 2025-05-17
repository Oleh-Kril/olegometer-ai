import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from visual_test.image_utils import phash_region, contains
import random

def _loc_similarity(p1, p2, sigma):
    """
    Gaussian fall‑off from 1 at distance 0 to 0 as distance → ∞.
    sigma controls how quickly it drops (≈ point where sim ≈ 0.61).
    """
    d = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
    return np.exp(-0.5 * (d / sigma) ** 2)

def visualize_matching_step(roiD, roiR, sim_hist, sim_ssim, score, title="", candidates=None):
    """Visualize a single matching step between two elements"""
    if candidates is None:
        candidates = [roiR]
    
    n_candidates = len(candidates)
    n_cols = min(4, n_candidates + 1)  # +1 for original element
    n_rows = (n_candidates + 1 + n_cols - 1) // n_cols  # Ceiling division
    
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        ax = np.array([ax])
    ax = ax.flatten()
    
    # Original design element
    ax[0].imshow(roiD, cmap='gray')
    ax[0].set_title('Design Element')
    
    # Show all candidates
    for i, cand in enumerate(candidates):
        ax[i + 1].imshow(cand, cmap='gray')
        if i == 0:  # First candidate is the current match
            ax[i + 1].set_title(f'Current Match\nHist={sim_hist:.3f}, SSIM={sim_ssim:.3f}\nScore={score:.3f}')
        else:
            ax[i + 1].set_title(f'Candidate {i}')
    
    # Hide unused subplots
    for i in range(n_candidates + 1, len(ax)):
        ax[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def visualize_candidates(roiD, candidates, distances, title=""):
    """Visualize the original element and all its hash-based candidates with their distances"""
    n_candidates = len(candidates)
    n_cols = min(4, n_candidates + 1)  # +1 for original element
    n_rows = (n_candidates + 1 + n_cols - 1) // n_cols  # Ceiling division
    
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        ax = np.array([ax])
    ax = ax.flatten()
    
    # Original design element
    ax[0].imshow(roiD, cmap='gray')
    ax[0].set_title('Design Element')
    
    # Show all candidates with their distances
    for i, (cand, dist) in enumerate(zip(candidates, distances)):
        ax[i + 1].imshow(cand, cmap='gray')
        ax[i + 1].set_title(f'Candidate {i}\nHash Distance: {dist}')
    
    # Hide unused subplots
    for i in range(n_candidates + 1, len(ax)):
        ax[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def match_elements(elems_D, elems_R, masked_D, masked_R,
                   hashes_D, hashes_R, constants, img_shape):
    """
    Compare histogram, SSIM **and** location similarity.
    img_shape = (H, W) of the screenshot – used to normalise distances.
    Elements are sorted by area in descending order (largest first).
    Children of unmatched parent elements are skipped.
    """
    # Sort elements by area in descending order
    elems_D = sorted(elems_D, key=lambda e: e['bbox'][2] * e['bbox'][3], reverse=True)
    elems_R = sorted(elems_R, key=lambda e: e['bbox'][2] * e['bbox'][3], reverse=True)
    
    matches_DR = {}
    unmatched_parents_D = set()  # Track unmatched parent elements
    H, W = img_shape
    diag = np.hypot(W, H)                          # image diagonal
    sigma = constants['LOCATION_SIGMA_FRAC'] * diag

    for i, eD in enumerate(elems_D):
        # Skip if this element is a child of an unmatched parent
        if any(contains(elems_D[p]['bbox'], eD['bbox']) for p in unmatched_parents_D):
            continue

        dists = [abs(hashes_D[i] - h) for h in hashes_R]
        # cand  = np.argsort(dists)[:constants['MAX_HASH_MATCHES']]

        cand = list(range(len(elems_R)))

        best, match_threshold = None, constants['MATCH_THRESHOLD']
        xD, yD, wD, hD = eD['bbox']
        roiD = masked_D[yD:yD + hD, xD:xD + wD]
        cD   = (xD + wD / 2, yD + hD / 2)          # centre of design element

        # Collect all candidate ROIs and their distances for visualization
        candidate_rois = []
        candidate_dists = []
        for j in cand:
            eR = elems_R[j]
            xR, yR, wR, hR = eR['bbox']
            roiR = masked_R[yR:yR + hR, xR:xR + wR]
            if roiD.size == 0 or roiR.size == 0:
                continue
            candidate_rois.append(roiR)
            candidate_dists.append(dists[j])

        # Visualize all candidates with their distances
        # visualize_candidates(
        #     roiD,
        #     candidate_rois,
        #     candidate_dists,
        #     f"Element D{i} and its hash-based candidates"
        # )

        for j in cand:
            eR = elems_R[j]
            xR, yR, wR, hR = eR['bbox']
            roiR = masked_R[yR:yR + hR, xR:xR + wR]
            if roiD.size == 0 or roiR.size == 0:
                continue

            # Skip if dimensions differ by more than 2x
            # if wR > 2 * wD or wD > 2 * wR or hR > 2 * hD or hD > 2 * hR:
            #     continue

            # --- appearance similarities -----------------------------------
            if roiD.shape != roiR.shape:
                roiR = cv2.resize(roiR, (roiD.shape[1], roiD.shape[0]))

            h1 = cv2.normalize(cv2.calcHist([roiD], [0], None, [64], [0, 256]),
                               None).ravel()
            h2 = cv2.normalize(cv2.calcHist([roiR], [0], None, [64], [0, 256]),
                               None).ravel()
            sim_hist = 1 - np.sum(np.abs(h1 - h2)) / len(h1)

            hmin, wmin = min(roiD.shape[0], roiR.shape[0]), \
                         min(roiD.shape[1], roiR.shape[1])
            sim_ssim = (ssim(roiD[:hmin, :wmin], roiR[:hmin, :wmin])
                        if hmin >= 7 and wmin >= 7 else 0)

            # --- location similarity --------------------------------------
            # cR = (xR + wR / 2, yR + hR / 2)
            # sim_loc = _loc_similarity(cD, cR, sigma)

            # --- combined score -------------------------------------------
            s = (constants['HISTOGRAM_WEIGHT'] * sim_hist +
                 constants['SSIM_WEIGHT']       * sim_ssim)

            if s > match_threshold:
                best, match_threshold = j, s

        if best is not None:
            matches_DR[i] = best
        else:
            if eD.get('children'):
                unmatched_parents_D.add(i)

    visualize_matching_result(masked_D, masked_R, elems_D, elems_R, matches_DR)
    return matches_DR


def visualize_matching_result(img_D, img_R, elems_D, elems_R, matches_DR):
    """
    Visualize element matching results.
    - Matched pairs: same random color on both images.
    - Unmatched: black bbox.
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(img_D, cmap='gray')
    ax[0].set_title('Design')
    ax[1].imshow(img_R, cmap='gray')
    ax[1].set_title('Real')

    # Assign a color for each match
    color_map = {}
    for i, j in matches_DR.items():
        color = tuple(random.random() for _ in range(3))
        color_map[i] = color
        color_map[j] = color  # Use same color for the pair

    # Draw bboxes for design elements
    for idx, elem in enumerate(elems_D):
        x, y, w, h = elem['bbox']
        if idx in matches_DR:
            color = color_map[idx]
        else:
            color = (0, 0, 0)  # black for unmatched
        ax[0].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, lw=2))

    # Draw bboxes for real elements
    matched_R = set(matches_DR.values())
    for idx, elem in enumerate(elems_R):
        x, y, w, h = elem['bbox']
        if idx in matched_R:
            color = color_map[idx]
        else:
            color = (0, 0, 0)  # black for unmatched
        ax[1].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, lw=2))

    plt.tight_layout()
    plt.show() 