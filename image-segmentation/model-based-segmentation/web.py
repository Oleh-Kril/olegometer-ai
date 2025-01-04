from datasets import load_dataset
import torch
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
ds = load_dataset("Zexanima/website_screenshots_image_dataset")

item = ds['valid'][0]
print(item)
# Convert bounding boxes from xywh to xyxy format
boxes_xywh = torch.tensor([obj['bbox'] for obj in item['objects']])
boxes_xyxy = box_convert(boxes_xywh, 'xywh', 'xyxy')

# Convert the image to a tensor and draw bounding boxes
image_tensor = pil_to_tensor(item['image'])
image_with_boxes_tensor = draw_bounding_boxes(
    image_tensor,
    boxes_xyxy,
    colors="red"  # Specify the color for the bounding boxes
)

# Convert the tensor back to a PIL image and display it
image_with_boxes_pil = to_pil_image(image_with_boxes_tensor)
image_with_boxes_pil.show()  # This will display the image with bounding boxes
