from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2, numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# 1) load your SAM-2 checkpoint via the registry
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_2b.pth")

# 2) configure generator
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=64,         # default 32 → more dense sampling
    pred_iou_thresh=0.30,       # default 0.88 → keep lower‐confidence masks
    stability_score_thresh=0.30,# default 0.92 → allow more “unstable” regions
    min_mask_region_area=50     # default 100 → include smaller islands
)

# 3) prepare image
image = cv2.imread("../../dataset/images/3a.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w = image_rgb.shape[:2]

# 4) generate masks
masks = mask_generator.generate(image_rgb)

# 5) build label map
label_map = np.zeros((h, w), dtype=np.int32)
for idx, m in enumerate(masks, start=1):
    label_map[m["segmentation"]] = idx

# 6) visualize
cmap = colors.ListedColormap(np.random.rand(len(masks) + 1, 3))
plt.figure(figsize=(8, 8))
plt.imshow(label_map, cmap=cmap)
plt.axis("off")
plt.title("SAM-2 Auto Masks (dense + low thresh)")
plt.show()
