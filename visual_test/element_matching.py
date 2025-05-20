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
                   hashes_D, hashes_R, constants, img_shape,
                   text_boxes_D=None, text_boxes_R=None):
    """
    Compare histogram, SSIM, (and optionally location).
    Returns a dict mapping ORIGINAL-D-index → ORIGINAL-R-index.
    """

    # 0) keep originals for later visualization
    orig_elems_D, orig_elems_R = elems_D, elems_R

    # 1) record original indices
    idxs_D = list(range(len(elems_D)))
    idxs_R = list(range(len(elems_R)))

    # 2) zip, sort by area descending
    def area(e): return e['bbox'][2] * e['bbox'][3]
    zipped_D = list(zip(elems_D, idxs_D, hashes_D))
    zipped_R = list(zip(elems_R, idxs_R, hashes_R))

    zipped_D.sort(key=lambda x: area(x[0]), reverse=True)
    zipped_R.sort(key=lambda x: area(x[0]), reverse=True)

    elems_D_s, idxs_D_s, hashes_D_s = zip(*zipped_D) if zipped_D else ([],[],[])
    elems_R_s, idxs_R_s, hashes_R_s = zip(*zipped_R) if zipped_R else ([],[],[])

    # 3) begin matching on sorted lists
    matches_s = {}                # sorted-index → sorted-index
    unmatched_parents = set()
    H, W = img_shape
    diag  = np.hypot(W, H)
    sigma = constants['LOCATION_SIGMA_FRAC'] * diag

    for i, eD in enumerate(elems_D_s):
        # skip children of unmatched parents
        if any(contains(elems_D_s[p]['bbox'], eD['bbox'])
               for p in unmatched_parents):
            continue

        # histogram distances for this design element
        dists = [abs(hashes_D_s[i] - hr) for hr in hashes_R_s]
        cand  = range(len(elems_R_s))

        # extract ROI for design
        xD,yD,wD,hD = eD['bbox']
        roiD = masked_D[yD:yD+hD, xD:xD+wD]
        if roiD.size == 0:
            continue

        # --- visualize all hash-based candidates ---
        candidate_rois  = []
        candidate_dists = []
        for j in cand:
            xR,yR,wR,hR = elems_R_s[j]['bbox']
            roiR = masked_R[yR:yR+hR, xR:xR+wR]
            if roiR.size == 0:
                continue
            candidate_rois.append(roiR)
            candidate_dists.append(dists[j])

        # visualize_candidates(
        #     roiD,
        #     candidate_rois,
        #     candidate_dists,
        #     f"Element D{idxs_D_s[i]}: hash candidates"
        # )

        # now find best match by appearance
        best, thr = None, constants['MATCH_THRESHOLD']
        for j in cand:
            xR,yR,wR,hR = elems_R_s[j]['bbox']
            roiR = masked_R[yR:yR+hR, xR:xR+wR]
            if roiR.size == 0:
                continue

            # resize to match shapes
            if roiD.shape != roiR.shape:
                roiR = cv2.resize(roiR, (roiD.shape[1], roiD.shape[0]))

            # histogram similarity
            h1 = cv2.normalize(cv2.calcHist([roiD],[0],None,[64],[0,256]), None).ravel()
            h2 = cv2.normalize(cv2.calcHist([roiR],[0],None,[64],[0,256]), None).ravel()
            sim_hist = 1 - np.sum(np.abs(h1 - h2)) / len(h1)

            # SSIM
            hmin, wmin = min(roiD.shape[0], roiR.shape[0]), min(roiD.shape[1], roiR.shape[1])
            sim_ssim = (ssim(roiD[:hmin, :wmin], roiR[:hmin, :wmin])
                        if hmin >= 7 and wmin >= 7 else 0)

            # combined score (location omitted here)
            s = (constants['HISTOGRAM_WEIGHT'] * sim_hist +
                 constants['SSIM_WEIGHT']       * sim_ssim)

            if s > thr:
                best, thr = j, s

        if best is not None:
            matches_s[i] = best
        else:
            if eD.get('children'):
                unmatched_parents.add(i)

    # 4) map sorted-indices back to original indices
    matches_DR = {
        idxs_D_s[i]: idxs_R_s[j]
        for i, j in matches_s.items()
    }

    # 5) final visualization on original ordering
    # visualize_matching_result(
    #     masked_D, masked_R,
    #     orig_elems_D, orig_elems_R,
    #     matches_DR
    # )

    return matches_DR


def visualize_matching_result(img_D, img_R, elems_D, elems_R, matches_DR):
    fig, (axD, axR) = plt.subplots(1, 2, figsize=(16, 8))
    axD.imshow(img_D, cmap='gray'); axD.set_title('Design')
    axR.imshow(img_R, cmap='gray'); axR.set_title('Real')

    # 1) Draw matched pairs in their own random color
    for i, j in matches_DR.items():
        color = tuple(random.random() for _ in range(3))
        xD, yD, wD, hD = elems_D[i]['bbox']
        xR, yR, wR, hR = elems_R[j]['bbox']
        axD.add_patch(plt.Rectangle((xD, yD), wD, hD,
                                    fill=False, edgecolor=color, lw=2))
        axR.add_patch(plt.Rectangle((xR, yR), wR, hR,
                                    fill=False, edgecolor=color, lw=2))

    # 2) Draw unmatched in black
    unmatched_D = set(range(len(elems_D))) - set(matches_DR.keys())
    unmatched_R = set(range(len(elems_R))) - set(matches_DR.values())
    for idx in unmatched_D:
        x, y, w, h = elems_D[idx]['bbox']
        axD.add_patch(plt.Rectangle((x, y), w, h,
                                    fill=False, edgecolor='black', lw=2))
    for idx in unmatched_R:
        x, y, w, h = elems_R[idx]['bbox']
        axR.add_patch(plt.Rectangle((x, y), w, h,
                                    fill=False, edgecolor='black', lw=2))

    plt.tight_layout()
    plt.show()