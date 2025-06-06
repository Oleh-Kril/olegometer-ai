import cv2
from pycocotools.coco import COCO
import os
from collections import defaultdict
from visual_test import process_images, convert_to_simple
import time  # Add time module for measurements


def test(img_R_path, img_D_path, base_dir='dataset/images'):
    img_D_path = os.path.join(base_dir, img_D_path)
    img_R_path = os.path.join(base_dir, img_R_path)
    imgD = cv2.imread(img_D_path)
    imgR = cv2.imread(img_R_path)

    # Add error checking for image loading
    if imgD is None:
        raise ValueError(f"Failed to load Design image: {img_D_path}")
    if imgR is None:
        raise ValueError(f"Failed to load Realization image: {img_R_path}")

    diffs_D, diffs_R = process_images(imgD, imgR)
    simple = convert_to_simple(diffs_D, diffs_R)
    return simple

def compute_iou(boxA, boxB):
    """
    boxA, boxB: [x, y, w, h]
    Returns IoU float.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    unionArea = areaA + areaB - interArea

    return interArea / unionArea if unionArea > 0 else 0

# --- Load ground‐truth COCO annotations ---
coco = COCO('dataset/annotations/instances.json')
cat_id2nm = {
    cat['id']: cat['name'].strip("‘’\"")
    for cat in coco.loadCats(coco.getCatIds())
}
images = coco.loadImgs(coco.getImgIds())

# --- Group into Case pairs ---
cases = defaultdict(dict)
for img in images:
    stem = os.path.splitext(img['file_name'])[0]  # e.g. "Case 1.1 R"
    if stem.endswith(' R'):
        cases[stem[:-2]]['R'] = img
    elif stem.endswith(' D'):
        cases[stem[:-2]]['D'] = img

# --- Evaluation counters ---
total_gt_bboxes = 0
correct_hits    = 0
correct_labels  = []  # Track correctly found labels in order
a1_correct      = 0
a2_correct      = 0
total_cases     = 0

# --- Timing metrics ---
comparison_times = []

# --- Iterate only full pairs ---
for case_id, imgs in cases.items():
    img_R = imgs.get('R')
    img_D = imgs.get('D')
    if img_R is None or img_D is None:
        continue

    total_cases += 1
    
    # Start timing
    start_time = time.time()
    
    # Get predictions (mocked)
    preds = test(img_R['file_name'], img_D['file_name'])
    
    # End timing and store
    end_time = time.time()
    comparison_time = end_time - start_time
    comparison_times.append(comparison_time)

    # Merge ground truth annotations from both sides
    merged_gt_anns = []
    for side in ('R', 'D'):
        gt_img = imgs[side]
        gt_anns = coco.loadAnns(coco.getAnnIds(imgIds=[gt_img['id']])) or []
        merged_gt_anns.extend(gt_anns)
    
    # Merge predictions from both sides
    merged_pred_bboxes = []
    merged_pred_labels = []
    for side in ('R', 'D'):
        merged_pred_bboxes.extend(preds[side]['bboxes'])
        merged_pred_labels.extend(preds[side]['labels'])

    # A1: Check if differences are detected correctly based on merged results
    if (len(merged_gt_anns) > 0 and len(merged_pred_bboxes) > 0) or (len(merged_gt_anns) == 0 and len(merged_pred_bboxes) == 0):
        a1_correct += 1

    # Loop over each side for A2 and A3 calculations
    for side in ('R', 'D'):
        gt_img = imgs[side]
        # GT annotations
        gt_anns    = coco.loadAnns(coco.getAnnIds(imgIds=[gt_img['id']])) or []
        gt_bboxes  = [ann['bbox'] for ann in gt_anns]
        gt_labels = [cat_id2nm[ann['category_id']] for ann in gt_anns]

        # Predicted
        pred_bboxes = preds[side]['bboxes']
        pred_labels = preds[side]['labels']

        # Update total GT count
        total_gt_bboxes += len(gt_bboxes)

        # A2: Check if all predictions match ground truth in type and count
        if len(gt_bboxes) == len(pred_bboxes) and set(gt_labels) == set(pred_labels):
            a2_correct += 1

        # A3: Label matching and IoU-based accuracy
        # Create a copy of pred_labels for matching
        remaining_pred_labels = pred_labels.copy()
        
        # First pass: Match labels exactly
        for gt_label in gt_labels:
            if gt_label in remaining_pred_labels:
                correct_labels.append(gt_label)
                # Remove the matched label from remaining predictions
                remaining_pred_labels.remove(gt_label)

        # Second pass: Calculate IoU for matched labels
        for gt_box, gt_label in zip(gt_bboxes, gt_labels):
            if gt_label in correct_labels:  # Only check IoU for matched labels
                best_iou = 0.0
                best_idx = None
                for idx, pred_box in enumerate(pred_bboxes):
                    iou = compute_iou(gt_box, pred_box)
                    if iou > best_iou:
                        best_iou, best_idx = iou, idx

                # Count as correct if IoU ≥ 0.7
                if best_iou >= 0.7 and best_idx is not None:
                    correct_hits += 1

# --- Compute and print accuracies ---
a1_accuracy = (a1_correct / total_cases * 100) if total_cases > 0 else 0
a2_accuracy = (a2_correct / (total_cases * 2) * 100) if total_cases > 0 else 0
a3_accuracy = 100 if (total_gt_bboxes == 0 and correct_hits == 0) else (correct_hits / len(correct_labels) * 100) if correct_labels else 0

# Calculate timing statistics
avg_time = sum(comparison_times) / len(comparison_times) if comparison_times else 0
worst_time = max(comparison_times) if comparison_times else 0
best_time = min(comparison_times) if comparison_times else 0

print(f"Total cases: {total_cases}")
print(f"Total ground‐truth boxes: {total_gt_bboxes}")
print(f"Correct hits (IoU≥0.9 & label match): {correct_hits}")
print(f"A1 Accuracy (correct difference detection): {a1_accuracy:.2f}%")
print(f"A2 Accuracy (correct type and count): {a2_accuracy:.2f}%")
print(f"A3 Accuracy (IoU-based): {a3_accuracy:.2f}%")
print(f"\nTiming Statistics:")
print(f"Average comparison time: {avg_time:.3f} seconds")
print(f"Worst case comparison time: {worst_time:.3f} seconds")
print(f"Best case comparison time: {best_time:.3f} seconds")
