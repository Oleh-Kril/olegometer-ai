import os
import cv2
from pycocotools.coco import COCO
from visual_test import process_images, convert_to_simple

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter
    return inter / union if union else 0.0

def run_single_case(case_name: str,
                    coco_file='dataset/annotations/instances.json',
                    img_root='dataset/images',
                    iou_thr=0.80):
    """
    Evaluate ONE case (e.g.  'Case 1.1') and return a dict with
    total GT boxes, correct hits and accuracy for that case.
    """
    coco = COCO(coco_file)
    cat_id2nm = {
        cat['id']: cat['name'].strip("‘’\"")  # strips the curly / straight quotes
        for cat in coco.loadCats(coco.getCatIds())
    }

    def find_record(suffix):
        fn = f"{case_name} {suffix}.png"
        # Get all image IDs first
        all_img_ids = coco.getImgIds()
        # Then filter for the specific filename
        matching_imgs = [img_id for img_id in all_img_ids 
                        if coco.loadImgs(img_id)[0]['file_name'] == fn]
        if not matching_imgs:
            raise FileNotFoundError(f"Image '{fn}' not listed in COCO json")
        return coco.loadImgs(matching_imgs)[0]

    img_rec_D = find_record('D')
    img_rec_R = find_record('R')

    imgD = cv2.imread(os.path.join(img_root, img_rec_D['file_name']))
    imgR = cv2.imread(os.path.join(img_root, img_rec_R['file_name']))
    if imgD is None or imgR is None:
        raise ValueError("Failed to load one of the images from disk")

    preds_dict = convert_to_simple(*process_images(imgD, imgR))

    def gt_for(img_rec):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_rec['id']])) or []
        boxes = [ann['bbox'] for ann in anns]
        labels = [cat_id2nm[ann['category_id']] for ann in anns]
        return boxes, labels

    gt_D_boxes, gt_D_labels = gt_for(img_rec_D)
    gt_R_boxes, gt_R_labels = gt_for(img_rec_R)

    def score_side(gt_boxes, gt_labels, pred_boxes, pred_labels):
        hits = 0
        for gbox, glab in zip(gt_boxes, gt_labels):
            best = -1
            best_idx = None
            for idx, pbox in enumerate(pred_boxes):
                iou = compute_iou(gbox, pbox)
                if iou > best:
                    best, best_idx = iou, idx
            if best >= iou_thr and best_idx is not None \
               and pred_labels[best_idx] == glab:
                hits += 1
        return hits, len(gt_boxes)

    hits_D, total_D = score_side(gt_D_boxes, gt_D_labels,
                                preds_dict['D']['bboxes'], preds_dict['D']['labels'])
    hits_R, total_R = score_side(gt_R_boxes, gt_R_labels,
                                preds_dict['R']['bboxes'], preds_dict['R']['labels'])

    total_hits = hits_D + hits_R
    total_gt = total_D + total_R
    acc = total_hits / total_gt * 100 if total_gt else 0

    return {
        'case': case_name,
        'gt_boxes': total_gt,
        'correct_hits': total_hits,
        'accuracy_%': round(acc, 2)
    }

if __name__ == "__main__":
    print(run_single_case("Case 1.3"))