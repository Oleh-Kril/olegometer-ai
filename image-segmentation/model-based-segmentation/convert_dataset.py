import os

from datasets import load_dataset

ds = load_dataset("mrtoy/mobile-ui-design")

# def convert_bbox_to_yolo_format(width, height, x_min, y_min, box_width, box_height):
#     x_center = (x_min + box_width / 2) / width
#     y_center = (y_min + box_height / 2) / height
#     norm_width = box_width / width
#     norm_height = box_height / height
#     return [x_center, y_center, norm_width, norm_height]


# for i, data in enumerate(ds['train']):
#     width = data['width']
#     height = data['height']
#     boxes = data['objects']['bbox']
#     print(i)
#
#     if i <= 1520:
#         with open(f"labels/val/{i}.txt", "w") as file:
#             for box in boxes:
#                 x_min, y_min, box_width, box_height = box
#                 yolo_box = convert_bbox_to_yolo_format(width, height, x_min, y_min, box_width, box_height)
#                 file.write(f"0 {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n")  # assuming class_id = 0

for i, data in enumerate(ds['train']):
    img = data['image']

    folder = "val" if i <= 1520 else "train"
    img_path = f"images/{folder}/{i}.jpg"
    img.save(img_path)
    print(i)
