import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_masks(mask, mask_closed, rect_mask):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(mask, cmap='gray')
    ax[0].set_title('Raw Difference Mask')
    ax[1].imshow(mask_closed, cmap='gray')
    ax[1].set_title('Closed Mask')
    ax[2].imshow(rect_mask, cmap='gray')
    ax[2].set_title('Rectangular Mask')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()

def create_rectangular_diff_mask(gray_D, gray_R, morphology_kernel_size):
    if gray_D.shape[0] != gray_R.shape[0]:
        h = max(gray_D.shape[0], gray_R.shape[0])
        tmpD = np.zeros((h, gray_D.shape[1]), np.uint8)
        tmpR = np.zeros((h, gray_R.shape[1]), np.uint8)
        tmpD[:gray_D.shape[0], :] = gray_D
        tmpR[:gray_R.shape[0], :] = gray_R
        gray_D, gray_R = tmpD, tmpR

    diff = cv2.absdiff(gray_D, gray_R)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (morphology_kernel_size, morphology_kernel_size)
    )
    mask_clear = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel2 = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (morphology_kernel_size*3, morphology_kernel_size*3)
    )
    mask_restored = cv2.morphologyEx(mask_clear, cv2.MORPH_CLOSE, kernel2)

    contours, _ = cv2.findContours(
        mask_restored,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    rect_mask = np.zeros_like(mask_restored)
    pad = 2
    height, width = rect_mask.shape
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width, x + w + pad)
        y2 = min(height, y + h + pad)
        cv2.rectangle(rect_mask, (x1, y1), (x2, y2), 255, thickness=-1)

    # visualize_masks(mask, mask_restored, rect_mask)

    return rect_mask 