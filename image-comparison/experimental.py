import cv2
import numpy as np


def segment_ui_elements(image: np.ndarray) -> list:
    """
    Segment the UI image into bounding boxes of UI components using image processing.
    Returns a list of bounding boxes [x, y, w, h] for each detected element.
    """
    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection to find component boundaries
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    # Dilate edges to close gaps between edge segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and collect bounding boxes for significant contours
    elements = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Heuristic filters to skip tiny or line-like regions (noise or separators)
        if w < 5 or h < 5:
            continue  # too small to be a UI element
        if w * h < 50:
            continue  # skip very small area components
        # (Optional) skip extremely thin lines (potentially just dividers)
        aspect_ratio = w / float(h)
        if aspect_ratio > 10 or aspect_ratio < 0.1:
            # This might be a long line or very tall thin line – likely not a standalone element
            continue
        elements.append((x, y, w, h))

    # Merge overlapping or nearby bounding boxes (to group parts of the same element)
    merged = True
    while merged:
        merged = False
        new_elements = []
        while elements:
            bx = elements.pop(0)
            x, y, w, h = bx
            x2, y2 = x + w, y + h
            # Try to merge with any other box that overlaps or is very close
            merged_with = None
            for i, b2 in enumerate(elements):
                xx, yy, ww, hh = b2
                xx2, yy2 = xx + ww, yy + hh
                # Check overlap or close proximity
                if not (x2 < xx - 5 or xx2 < x - 5 or y2 < yy - 5 or yy2 < y - 5):
                    # Merge the two boxes
                    nx = min(x, xx)
                    ny = min(y, yy)
                    nx2 = max(x2, xx2)
                    ny2 = max(y2, yy2)
                    nw, nh = nx2 - nx, ny2 - ny
                    bx = (nx, ny, nw, nh)
                    # Remove b2 and mark merged
                    elements.pop(i)
                    merged = True
                    break
            new_elements.append(bx)
        elements = new_elements
    return elements

def compute_iou(box1, box2) -> float:
    """Compute Intersection-over-Union of two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    # Coordinates of intersections
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1+w1, x2+w2)
    inter_y2 = min(y1+h1, y2+h2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0  # no overlap
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = w1 * h1
    area2 = w2 * h2
    iou = inter_area / float(area1 + area2 - inter_area)
    return iou

def match_elements(ref_elements: list, test_elements: list) -> tuple:
    """
    Match UI elements between reference and test images using IoU overlap.
    Returns (matches, unmatched_ref, unmatched_test):
      - matches: list of (ref_index, test_index) pairs
      - unmatched_ref: indices in ref_elements with no match (missing in test)
      - unmatched_test: indices in test_elements with no match (extra in test)
    """
    matches = []
    used_test = set()
    # Match each reference element to the best overlapping test element
    for i, ref_box in enumerate(ref_elements):
        best_j, best_iou = None, 0.0
        for j, test_box in enumerate(test_elements):
            if j in used_test:
                continue  # already matched this test element
            iou = compute_iou(ref_box, test_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        # If overlap is above a threshold, consider it a match
        if best_j is not None and best_iou > 0.3:  # threshold can be tuned
            matches.append((i, best_j))
            used_test.add(best_j)
    # Any reference indices not matched are missing in the test image
    matched_ref_indices = {i for i, _ in matches}
    unmatched_ref = [i for i in range(len(ref_elements)) if i not in matched_ref_indices]
    # Any test indices not matched are extra in the test image
    matched_test_indices = {j for _, j in matches}
    unmatched_test = [j for j in range(len(test_elements)) if j not in matched_test_indices]
    return matches, unmatched_ref, unmatched_test


def analyze_differences(ref_image: np.ndarray, test_image: np.ndarray,
                        ref_elements: list, test_elements: list,
                        matches: list, unmatched_ref: list, unmatched_test: list) -> list:
    """
    Analyze matched elements for differences and list issues for unmatched elements.
    Returns a list of difference records, each a dict with:
      - 'bbox': (x, y, w, h) of the element (in the test image's coordinate space for visualization)
      - 'issue': description of the discrepancy
    """
    differences = []
    # 1. Missing elements (present in ref, missing in test)
    for i in unmatched_ref:
        x, y, w, h = ref_elements[i]
        differences.append({
            "bbox": (x, y, w, h),
            "issue": "Element missing in implementation (expected at {},{} size {}x{})".format(x, y, w, h)
        })
    # 2. Extra elements (present in test, not in ref)
    for j in unmatched_test:
        x, y, w, h = test_elements[j]
        differences.append({
            "bbox": (x, y, w, h),
            "issue": "Extra element in implementation (at {},{} size {}x{})".format(x, y, w, h)
        })
    # 3. Differences in matched elements
    for (i, j) in matches:
        ref_box = ref_elements[i]
        test_box = test_elements[j]
        xr, yr, wr, hr = ref_box
        xt, yt, wt, ht = test_box
        # Size check
        size_diff_width = abs(wr - wt)
        size_diff_height = abs(hr - ht)
        if size_diff_width > 3 or size_diff_height > 3:  # tolerance of 3px
            differences.append({
                "bbox": (xt, yt, wt, ht),
                "issue": f"Size mismatch for element (expected {wr}x{hr}, got {wt}x{ht})"
            })
        # Position (padding) check - check if the element moved by more than a few pixels
        pos_diff = np.sqrt((xr - xt)**2 + (yr - yt)**2)
        if pos_diff > 5:
            differences.append({
                "bbox": (xt, yt, wt, ht),
                "issue": f"Position offset for element (design at ({xr},{yr}), impl at ({xt},{yt}))"
            })
        # Color check - compare average color in the region (could also use histogram or SSIM)
        ref_region = ref_image[yr:yr+hr, xr:xr+wr]
        test_region = test_image[yt:yt+ht, xt:xt+wt]
        # Compute mean color difference (in BGR)
        ref_mean = cv2.mean(cv2.cvtColor(ref_region, cv2.COLOR_BGR2HSV))  # using HSV for color comparison
        test_mean = cv2.mean(cv2.cvtColor(test_region, cv2.COLOR_BGR2HSV))
        # Difference in mean (ignoring alpha channel from cv2.mean)
        color_diff = np.linalg.norm(np.array(ref_mean[:3]) - np.array(test_mean[:3]))
        if color_diff > 20:  # threshold for color difference (empirical)
            differences.append({
                "bbox": (xt, yt, wt, ht),
                "issue": "Color difference in element (design vs. implementation colors vary)"
            })
        # (Future extension) Text/Font check can be added here with OCR and font analysis.
    return differences


def draw_differences(ref_image: np.ndarray, test_image: np.ndarray, differences: list) -> np.ndarray:
    """
    Create an output image visualizing all differences.
    We draw bounding boxes on a copy of the test_image (and reference if needed) to highlight issues.
    Returns an image (numpy array) with highlighted differences.
    """
    # Make copies to draw on
    out_ref = ref_image.copy()
    out_test = test_image.copy()
    # Colors for different issue types
    color_missing = (255, 0, 0)  # Blue for missing elements (drawn on reference image)
    color_extra = (0, 165, 255)  # Orange for extra elements (drawn on test image)
    color_diff = (0, 0, 255)  # Red for mismatched elements (on test image)

    for diff in differences:
        x, y, w, h = diff["bbox"]
        issue_text = diff["issue"]
        if "missing" in issue_text:
            # Draw on reference image where something is expected but missing in test
            cv2.rectangle(out_ref, (x, y), (x + w, y + h), color_missing, 2)
            cv2.putText(out_ref, "Missing", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_missing, 2)
        elif "Extra" in issue_text:
            # Draw on test image where an extra element is present
            cv2.rectangle(out_test, (x, y), (x + w, y + h), color_extra, 2)
            cv2.putText(out_test, "Extra", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_extra, 2)
        else:
            # Other differences (size, color, position) on test image
            cv2.rectangle(out_test, (x, y), (x + w, y + h), color_diff, 2)
            cv2.putText(out_test, "Diff", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_diff, 2)
    # Here we could combine out_ref and out_test side by side or overlay for final visualization.
    # For simplicity, return the test image with differences (out_test).
    return out_test

# Example usage (assuming ref_path and test_path are file paths to the images):
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# -----------------------------------------------
# 1. Display function using Matplotlib
def display_image_with_mask(img, title="Image with Mask"):
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


# -----------------------------------------------
# 2. Two separate difference masks:
def draw_differences_two_masks(ref_image, test_image, differences):
    """
    Draw two masks:
      - ref_mask: drawn on the reference image showing only missing elements
      - test_mask: drawn on the test image showing extra and other differences
    """
    ref_mask = ref_image.copy()
    test_mask = test_image.copy()
    color_missing = (255, 0, 0)  # Blue for missing elements
    color_diff = (0, 0, 255)  # Red for other differences
    color_extra = (0, 165, 255)  # Orange for extra elements

    for diff in differences:
        x, y, w, h = diff["bbox"]
        issue_text = diff["issue"]
        if "missing" in issue_text.lower():
            cv2.rectangle(ref_mask, (x, y), (x + w, y + h), color_missing, 2)
            cv2.putText(ref_mask, "Missing", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_missing, 2)
        else:
            if "extra" in issue_text.lower():
                cv2.rectangle(test_mask, (x, y), (x + w, y + h), color_extra, 2)
                cv2.putText(test_mask, "Extra", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_extra, 2)
            else:
                cv2.rectangle(test_mask, (x, y), (x + w, y + h), color_diff, 2)
                cv2.putText(test_mask, "Diff", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_diff, 2)
    return ref_mask, test_mask


# -----------------------------------------------
# 3. Display segmentation results using Matplotlib
def show_segmentation_results(image, elements, title="Segmentation Result"):
    """
    Display the image with segmentation bounding boxes.
    """
    img_seg = image.copy()
    for (x, y, w, h) in elements:
        cv2.rectangle(img_seg, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_seg, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


# -----------------------------------------------
# 4. New matching using SSIM instead of IoU
def match_elements_ssim(ref_image, test_image, ref_elements, test_elements, resize_dim=(64, 64), ssim_threshold=0.8):
    """
    Match UI elements using SSIM.
    Returns (matches, unmatched_ref, unmatched_test) where:
      - matches: list of (ref_index, test_index) pairs
      - unmatched_ref: indices in ref_elements with no match (missing in test)
      - unmatched_test: indices in test_elements with no match (extra in test)
    """
    matches = []
    used_test = set()
    # Convert images to grayscale for patch extraction
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    for i, (x, y, w, h) in enumerate(ref_elements):
        ref_patch = ref_gray[y:y + h, x:x + w]
        try:
            ref_patch_resized = cv2.resize(ref_patch, resize_dim)
        except Exception:
            continue
        best_j, best_ssim = None, -1
        for j, (xt, yt, wt, ht) in enumerate(test_elements):
            if j in used_test:
                continue
            test_patch = test_gray[yt:yt + ht, xt:xt + wt]
            try:
                test_patch_resized = cv2.resize(test_patch, resize_dim)
            except Exception:
                continue
            score, _ = ssim(ref_patch_resized, test_patch_resized, full=True)
            if score > best_ssim:
                best_ssim = score
                best_j = j
        if best_j is not None and best_ssim >= ssim_threshold:
            matches.append((i, best_j))
            used_test.add(best_j)

    matched_ref_indices = {i for i, _ in matches}
    unmatched_ref = [i for i in range(len(ref_elements)) if i not in matched_ref_indices]
    matched_test_indices = {j for _, j in matches}
    unmatched_test = [j for j in range(len(test_elements)) if j not in matched_test_indices]
    return matches, unmatched_ref, unmatched_test


def display_matching_pairs(ref_image, test_image, ref_elements, test_elements, matches):
    """
    Display matching pairs of UI elements side by side.
    Lines connect the centers of matched bounding boxes.
    """
    # Create a combined image with ref_image on left and test_image on right
    height = max(ref_image.shape[0], test_image.shape[0])
    width = ref_image.shape[1] + test_image.shape[1]
    combined = np.zeros((height, width, 3), dtype=np.uint8)
    combined[:ref_image.shape[0], :ref_image.shape[1]] = ref_image
    combined[:test_image.shape[0], ref_image.shape[1]:] = test_image

    for (i, j) in matches:
        rx, ry, rw, rh = ref_elements[i]
        tx, ty, tw, th = test_elements[j]
        center_ref = (int(rx + rw / 2), int(ry + rh / 2))
        center_test = (int(tx + tw / 2) + ref_image.shape[1], int(ty + th / 2))
        # Draw bounding boxes in green
        cv2.rectangle(combined, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        cv2.rectangle(combined, (tx + ref_image.shape[1], ty), (tx + ref_image.shape[1] + tw, ty + th), (0, 255, 0), 2)
        # Draw a line connecting the centers
        cv2.line(combined, center_ref, center_test, (255, 0, 0), 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.title("Matching Pairs of UI Elements")
    plt.axis("off")
    plt.show()


# -----------------------------------------------
# 5. Pretrained UI element detector (placeholder)
def segment_ui_elements_detector(image: np.ndarray) -> list:
    """
    Segment UI elements using a pretrained detector.
    Replace this placeholder with actual model inference code (e.g., using Detectron2 or a Faster R-CNN model).
    For now, we use the heuristic segmentation function (assumed to be defined elsewhere).
    """
    return segment_ui_elements(image)  # Call your existing segmentation function


# -----------------------------------------------
# Example usage (for testing):
import pytesseract


def segment_text_areas(image: np.ndarray) -> list:
    """
    Use pytesseract to detect text areas.
    Returns a list of dicts with keys:
      - 'bbox': (x, y, w, h)
      - 'text': recognized text string
    """
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n_boxes = len(data['level'])
    text_boxes = []
    for i in range(n_boxes):
        text = data['text'][i].strip()
        if text == "":
            continue
        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]
        text_boxes.append({"bbox": (x, y, w, h), "text": text})
    return text_boxes


def compare_text_areas(ref_text_boxes: list, test_text_boxes: list) -> list:
    """
    Compare text areas between the reference and test images.
    For each text box in the reference, attempt to match a corresponding text box
    in the test image using IoU. If found, compare the recognized text (case-insensitive).
    Returns a list of differences (each with bbox and issue description).
    """

    def compute_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = w1 * h1
        area2 = w2 * h2
        return inter_area / float(area1 + area2 - inter_area)

    differences = []
    used_test = set()
    # For each text box in the reference image, find the best matching text box in the test image.
    for i, ref_box in enumerate(ref_text_boxes):
        ref_bbox = ref_box["bbox"]
        ref_text = ref_box["text"].strip().lower()
        best_j = None
        best_iou = 0.0
        for j, test_box in enumerate(test_text_boxes):
            if j in used_test:
                continue
            iou = compute_iou(ref_bbox, test_box["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j is not None and best_iou > 0.3:
            test_text = test_text_boxes[best_j]["text"].strip().lower()
            if ref_text != test_text:
                differences.append({
                    "bbox": ref_bbox,
                    "issue": f"Text mismatch: expected '{ref_text}' but got '{test_text}'"
                })
            used_test.add(best_j)
        else:
            differences.append({
                "bbox": ref_bbox,
                "issue": f"Text missing in test image: '{ref_text}'"
            })

    # Flag extra text areas in test image that were not matched.
    for j, test_box in enumerate(test_text_boxes):
        if j not in used_test:
            differences.append({
                "bbox": test_box["bbox"],
                "issue": f"Extra text in test image: '{test_box['text'].strip()}'"
            })

    return differences


def filter_text_from_ui_elements(ui_elements: list, text_boxes: list, overlap_threshold=0.3) -> list:
    """
    Remove UI element bounding boxes that overlap significantly with any detected text box.
    Returns a filtered list of UI element bounding boxes.
    """

    def compute_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = w1 * h1
        area2 = w2 * h2
        return inter_area / float(area1 + area2 - inter_area)

    filtered = []
    for ui_box in ui_elements:
        overlap = False
        for text_box in text_boxes:
            if compute_iou(ui_box, text_box["bbox"]) > overlap_threshold:
                overlap = True
                break
        if not overlap:
            filtered.append(ui_box)
    return filtered


# Example usage integration:
if __name__ == "__main__":
    # Load images
    ref_image = cv2.imread("../dataset/demo a.png")
    test_image = cv2.imread("../dataset/demo b.png")

    # 1. Segment text areas in both images.
    ref_text_boxes = segment_text_areas(ref_image)
    test_text_boxes = segment_text_areas(test_image)

    # Optionally, display text segmentation results using Matplotlib.
    import matplotlib.pyplot as plt


    def draw_text_boxes(image, text_boxes, title="Text Areas"):
        img_copy = image.copy()
        for tb in text_boxes:
            x, y, w, h = tb["bbox"]
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img_copy, tb["text"], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.show()


    draw_text_boxes(ref_image, ref_text_boxes, title="Reference Image Text Areas")
    draw_text_boxes(test_image, test_text_boxes, title="Test Image Text Areas")

    # 2. Compare recognized text between the images.
    text_differences = compare_text_areas(ref_text_boxes, test_text_boxes)
    print("Text differences found:", text_differences)

    # 3. Main UI segmentation (using the pretrained detector or heuristic function)
    # and then filtering out text regions.
    ref_ui_elements = segment_ui_elements_detector(ref_image)
    test_ui_elements = segment_ui_elements_detector(test_image)

    # Filter out UI elements overlapping with text areas.
    ref_ui_elements_filtered = filter_text_from_ui_elements(ref_ui_elements, ref_text_boxes)
    test_ui_elements_filtered = filter_text_from_ui_elements(test_ui_elements, test_text_boxes)

    # Optionally, display the filtered UI segmentation results.
    show_segmentation_results(ref_image, ref_ui_elements_filtered, title="Reference UI Elements (Text Removed)")
    show_segmentation_results(test_image, test_ui_elements_filtered, title="Test UI Elements (Text Removed)")

    # Continue with matching and further analysis on ref_ui_elements_filtered and test_ui_elements_filtered...


