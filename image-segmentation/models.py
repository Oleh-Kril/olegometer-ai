#!/usr/bin/env python3
"""
compare_segmentation_seq.py

Sequentially runs multiple segmentation_models.pytorch architectures
on one image, logging each result with bounding‐boxes drawn around
the predicted segments.
"""

import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import math
import os

# ---- CONFIG ----
IMAGE_PATH    = '../dataset/images/3a.png'
OUTPUT_DIR    = './'
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
ENCODER       = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
MIN_CONTOUR_AREA = 1000   # filter out tiny regions

MODELS = [
    ('UNet++',    smp.UnetPlusPlus),
    ('MAnet',     smp.MAnet),
    ('Linknet',   smp.Linknet),
    ('FPN',       smp.FPN),
    ('PSPNet',    smp.PSPNet),
    ('PAN',       smp.PAN),
    ('DeepLabV3+',smp.DeepLabV3Plus),
    ('UPerNet',   smp.UPerNet),
    ('Segformer', smp.Segformer),
    ('DPT',       smp.DPT),
]

def load_image(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read '{path}'")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0

def pad_to_divisible(img, div=32):
    h, w, _ = img.shape
    new_h = math.ceil(h/div)*div
    new_w = math.ceil(w/div)*div
    pad_h, pad_w = new_h - h, new_w - w
    tmp = (img*255).astype(np.uint8)
    padded = cv2.copyMakeBorder(tmp, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return padded.astype(np.float32)/255.0, (h, w)

def run_model(img_rgb, ModelClass):
    padded, (oh, ow) = pad_to_divisible(img_rgb, div=32)
    model = ModelClass(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1, activation=None
    ).to(DEVICE)
    model.eval()
    preprocess_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    x = preprocess_fn(padded.astype(np.float32))
    x = x.transpose(2,0,1)[None]  # 1×C×H×W
    x = torch.from_numpy(x).to(DEVICE)
    with torch.no_grad():
        logits = model(x.float())
        mask = torch.sigmoid(logits)[0,0].cpu().numpy()
    return mask[:oh, :ow]

def draw_bboxes(img_rgb, mask, thresh=0.5, min_area=MIN_CONTOUR_AREA):
    """
    Draws green bounding boxes on a copy of img_rgb
    around each connected region in mask > thresh.
    """
    vis = (img_rgb * 255).astype(np.uint8).copy()
    # binarize mask
    bw = (mask > thresh).astype(np.uint8) * 255
    # find contours
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
    return vis

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img = load_image(IMAGE_PATH)

    for name, ModelClass in MODELS:
        print(f"[→] Starting {name}…")
        try:
            mask = run_model(img, ModelClass)
            vis  = draw_bboxes(img, mask)
            # show window
            cv2.imshow(name, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyWindow(name)
            # optionally save:
            out_path = os.path.join(OUTPUT_DIR, f"bboxes_{name}.png")
            cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print(f"[✔] {name} done — saved to {out_path}")
        except Exception as e:
            print(f"[✖] {name} failed: {e}")

if __name__ == "__main__":
    main()
