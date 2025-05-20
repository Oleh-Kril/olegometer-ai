import easyocr
import numpy as np

from visual_test.custom_types import DiffType
from visual_test.image_utils import contains
import matplotlib.pyplot as plt
import cv2
import torch

def filter_nested_missing_elements(diffs):
    missing = [d for d in diffs
               if d['type'] == 'ELEMENT_MISSING_ON_WEBSITE_PAGE']
    keep = []
    for i, d in enumerate(missing):
        x1, y1, w1, h1 = d['bbox']
        contained = False
        for j, other in enumerate(missing):
            if i == j:
                continue
            x2, y2, w2, h2 = other['bbox']
            if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                contained = True
                break
        if not contained:
            keep.append(d)
    return keep + [d for d in diffs if d['type'] != 'ELEMENT_MISSING_ON_WEBSITE_PAGE']

def closest(b, cont, axis, w_img, h_img):
    if cont is None:
        return min(b[0], w_img - (b[0] + b[2])) if axis == 'x' else min(b[1], h_img - (b[1] + b[3]))
    px, py, pw, ph = cont
    return min(b[0] - px, px + pw - (b[0] + b[2])) if axis == 'x' else min(b[1] - py, py + ph - (b[1] + b[3]))

# in visual_test/comparison.py

def visualize_parent_child_relationships(img_D, img_R, eD, eR, parent_D, parent_R):
    """
    Visualize parent-child relationships on both design and real images.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Convert BGR to RGB for display
    img_D_rgb = cv2.cvtColor(img_D, cv2.COLOR_BGR2RGB)
    img_R_rgb = cv2.cvtColor(img_R, cv2.COLOR_BGR2RGB)

    # Display images
    ax[0].imshow(img_D_rgb)
    ax[0].set_title('Design')
    ax[1].imshow(img_R_rgb)
    ax[1].set_title('Real')

    # Draw element and parent on design image
    x, y, w, h = eD['bbox']
    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='blue', linewidth=2)
    ax[0].add_patch(rect)
    ax[0].text(x, y-5, 'Element', color='blue', fontsize=8,
              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    if parent_D is not None:
        px, py, pw, ph = parent_D
        parent_rect = plt.Rectangle((px, py), pw, ph, fill=False, edgecolor='red', linewidth=2, linestyle='--')
        ax[0].add_patch(parent_rect)
        ax[0].text(px, py-5, 'Parent', color='red', fontsize=8,
                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Draw element and parent on real image
    x, y, w, h = eR['bbox']
    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='blue', linewidth=2)
    ax[1].add_patch(rect)
    ax[1].text(x, y-5, 'Element', color='blue', fontsize=8,
              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    if parent_R is not None:
        px, py, pw, ph = parent_R
        parent_rect = plt.Rectangle((px, py), pw, ph, fill=False, edgecolor='red', linewidth=2, linestyle='--')
        ax[1].add_patch(parent_rect)
        ax[1].text(px, py-5, 'Parent', color='red', fontsize=8,
                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    plt.tight_layout()
    plt.show()

def compare_positions(eD, eR, parent_D, parent_R, masked_D, masked_R, constants, wrong_pos_seen):
    """
    Compare element positions, now relative to each side's container.
    parent_D: bbox of the container in the design image (or None)
    parent_R: bbox of the container in the real image   (or None)
    """
    # Visualize parent-child relationships before comparison
    # visualize_parent_child_relationships(masked_D, masked_R, eD, eR, parent_D, parent_R)

    dx = eR['bbox'][0] - eD['bbox'][0]
    dy = eR['bbox'][1] - eD['bbox'][1]
    sx = abs(dx) > constants['POSITION_THRESHOLD']
    sy = abs(dy) > constants['POSITION_THRESHOLD']

    # no significant move → no diff
    if not (sx or sy):
        return None

    # avoid duplicate reporting per element
    key = tuple(eD['bbox'])
    if key in wrong_pos_seen:
        return None
    wrong_pos_seen.add(key)

    # Determine if this is a container-level or page-level difference
    is_container_level = parent_D is not None
    container_key = parent_D if is_container_level else 'page'

    # Create the position difference record
    rec = {
        'bbox': eR['bbox'],
        'type': DiffType.WRONG_POSITION_WITHIN if is_container_level else DiffType.WRONG_POSITION,
        'container_key': container_key,
        'position': (eR['bbox'][0], eR['bbox'][1])  # Store position for sorting
    }

    # helper to pick distances against the correct container
    def dist(b, cont, axis, img_w, img_h):
        if cont is None:
            return min(b[0], img_w - (b[0] + b[2])) if axis=='x' else \
                   min(b[1], img_h - (b[1] + b[3]))
        px, py, pw, ph = cont
        if axis == 'x':
            return min(b[0] - px, px+pw - (b[0]+b[2]))
        else:
            return min(b[1] - py, py+ph - (b[1]+b[3]))

    if sx:
        rec['OriginalDistanceX'] = dist(eD['bbox'], parent_D,
                                        'x', masked_D.shape[1], masked_D.shape[0])
        rec['CurrentDistanceX']  = dist(eR['bbox'], parent_R,
                                        'x', masked_R.shape[1], masked_R.shape[0])
    if sy:
        rec['OriginalDistanceY'] = dist(eD['bbox'], parent_D,
                                        'y', masked_D.shape[1], masked_D.shape[0])
        rec['CurrentDistanceY']  = dist(eR['bbox'], parent_R,
                                        'y', masked_R.shape[1], masked_R.shape[0])

    return rec

def filter_position_differences(diffs, elems_R=None):
    """
    Filter position differences to keep only the single most top-left one at each level.
    Levels are:
    1. Page level (WRONG_POSITION)
    2. Container levels (WRONG_POSITION_WITHIN) - one per container
    """
    if not diffs:
        return diffs

    # Separate position differences by type
    wrong_positions = [d for d in diffs if d['type'] == DiffType.WRONG_POSITION]
    wrong_positions_within = [d for d in diffs if d['type'] == DiffType.WRONG_POSITION_WITHIN]
    other_diffs = [d for d in diffs if d['type'] not in [DiffType.WRONG_POSITION, DiffType.WRONG_POSITION_WITHIN]]

    filtered_diffs = []

    # Handle page-level position differences
    if wrong_positions:
        # Sort by position (x + y)
        sorted_positions = sorted(wrong_positions, key=lambda d: d['position'][0] + d['position'][1])
        filtered_diffs.append(sorted_positions[0])  # Keep only the most top-left one

    # Handle container-level position differences
    if wrong_positions_within and elems_R is not None:
        # Group differences by their container
        container_diffs = {}
        for diff in wrong_positions_within:
            # Convert container key to tuple if it's a list
            container_key = tuple(diff['container_key']) if isinstance(diff['container_key'], list) else diff['container_key']
            if container_key not in container_diffs:
                container_diffs[container_key] = []
            container_diffs[container_key].append(diff)

        # Process each container's differences
        for container_key, container_diff_list in container_diffs.items():
            # Sort differences within this container by position
            sorted_container_diffs = sorted(
                container_diff_list,
                key=lambda d: d['position'][0] + d['position'][1]
            )
            filtered_diffs.append(sorted_container_diffs[0])  # Keep only the most top-left one

    # Add back non-position differences
    filtered_diffs.extend(other_diffs)

    return filtered_diffs

def compare_sizes(eD, eR, constants):
    w0, h0 = eD['bbox'][2:4]
    w1, h1 = eR['bbox'][2:4]

    # Calculate percentage differences
    width_diff_percent = abs(w0 - w1) / w0 * 100
    height_diff_percent = abs(h0 - h1) / h0 * 100

    if width_diff_percent > constants['SIZE_THRESHOLD'] or height_diff_percent > constants['SIZE_THRESHOLD']:
        return {
            'type': DiffType.WRONG_SIZE,
            'bbox': eR['bbox'],
            'OriginalWidth': w0,
            'CurrentWidth': w1,
            'OriginalHeight': h0,
            'CurrentHeight': h1,
            'WidthDiffPercent': width_diff_percent,
            'HeightDiffPercent': height_diff_percent
        }
    return None

def visualize_masked_image(img: np.ndarray, mask: np.ndarray, title: str = "Masked Image"):
    """
    Visualize the image with masked areas blacked out.
    """
    # Create a copy of the image
    masked_img = img.copy()
    
    # Ensure mask is binary
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (mask > 0).astype(np.uint8) * 255
    
    # Apply mask
    masked_img = cv2.bitwise_and(masked_img, masked_img, mask=mask)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def detect_text_full_image(
    img: np.ndarray,
    reader: easyocr.Reader = None,
    mask: np.ndarray = None,
    *,
    scale: float = 1.5,
    clahe_clip: float = 2.0,
    low_text: float = 0.4,
    link_threshold: float = 0.4,
    width_ths: float = 0.7,
    height_ths: float = 0.7,
    min_confidence: float = 0.5
) -> list[dict]:
    # return []
    """
    Detect text in a GUI screenshot using EasyOCR, with preprocessing
    for better contrast and resolution.

    Args:
      img             : BGR or grayscale image as numpy array.
      reader          : Pre-initialized EasyOCR Reader.
      mask            : Binary mask where 0 indicates areas to ignore (will be blacked out).
      scale           : Upscale factor for the image.
      clahe_clip      : Clip limit for CLAHE contrast enhancement.
      low_text        : EasyOCR low_text parameter (prob threshold for text).
      link_threshold  : EasyOCR link_threshold parameter (for linking components).
      width_ths       : EasyOCR width threshold for grouping.
      height_ths      : EasyOCR height threshold for grouping.
      min_confidence  : Filter out detections below this probability.

    Returns:
      A list of dicts: { 'bbox': [x, y, w, h], 'text': str, 'confidence': float }.
    """
    # 1. Initialize reader if needed
    if reader is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        use_gpu = True if device == 'mps' else False
        reader = easyocr.Reader(['en', 'uk'], gpu=use_gpu)

    # 2. Ensure grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 3. Apply mask if provided
    if mask is not None:
        # Visualize masked image before processing
        # visualize_masked_image(img, mask, "Image with Masked Areas")
        
        # Ensure mask is binary
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask > 0).astype(np.uint8) * 255
        # Black out masked areas
        gray = cv2.bitwise_and(gray, gray, mask=mask)

    # 4. Contrast enhancement with CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 5. Upscale image
    h0, w0 = enhanced.shape
    w1, h1 = int(w0 * scale), int(h0 * scale)
    resized = cv2.resize(enhanced, (w1, h1), interpolation=cv2.INTER_LINEAR)

    # 6. Optional morphological opening to remove speckle noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(resized, cv2.MORPH_OPEN, kernel, iterations=1)

    # 7. Text detection & recognition
    #    We request detail=1 so we get bounding boxes + text + confidence
    results = reader.readtext(
        cleaned,
        detail=1,
        paragraph=False,
        low_text=low_text,
        link_threshold=link_threshold,
        width_ths=width_ths,
        height_ths=height_ths
    )

    # 8. Rescale boxes back to original image size and filter by confidence
    text_boxes = []
    for bbox, text, prob in results:
        if prob < min_confidence:
            continue

        # bbox: list of 4 points [[x1,y1], ...]
        (x1, y1), (_, _), (x2, y2), (_, _) = bbox
        # scale back
        x1o, y1o = int(x1 / scale), int(y1 / scale)
        x2o, y2o = int(x2 / scale), int(y2 / scale)
        w, h = x2o - x1o, y2o - y1o

        # Skip if the box is completely in masked area
        if mask is not None:
            box_mask = mask[y1o:y2o, x1o:x2o]
            if np.all(box_mask == 0):
                continue

        text_boxes.append({
            'bbox': [x1o, y1o, w, h],
            'text': text,
            'confidence': float(prob)
        })

    return text_boxes

def visualize_text_detection(img, text_boxes, title="Text Detection"):
    """
    Visualize text detection results on the image.
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    for box in text_boxes:
        x, y, w, h = box['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=2)
        plt.gca().add_patch(rect)
        plt.text(x, y-5, box['text'], color='green', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def compare_text(eD, eR, text_boxes_D, text_boxes_R):
    """
    Compare text between two elements using pre-detected text boxes.
    """
    # Get text from intersecting boxes
    tD = " ".join(box['text'] for box in text_boxes_D 
                 if contains(eD['bbox'], box['bbox'])).strip()
    tR = " ".join(box['text'] for box in text_boxes_R 
                 if contains(eR['bbox'], box['bbox'])).strip()
    
    if tD and tR and tD != tR:
        return {
            'type': DiffType.WRONG_COPY,
            'bbox': eR['bbox'],
            'OriginalCopy': tD,
            'CurrentCopy': tR
        }
    return None 