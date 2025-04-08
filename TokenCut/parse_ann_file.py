import json
from PIL import Image
import os

ann_file = '/media/u6566739/c42f350c-c236-4250-8574-4dcf0809a9e5/VOCdevkit/VOC2012/trainval_tokencut_ann'
root = '/media/u6566739/c42f350c-c236-4250-8574-4dcf0809a9e5/VOCdevkit/VOC2012/JPEGImages'
with open(ann_file, 'r') as f:
    dict = json.load(f)

count = 0
for i, (img, box) in enumerate(zip(dict["image_files"],dict["bboxes"])):
    print(f'Processing {i}/{len(dict["bboxes"])}')
    image_path = os.path.join(root, img)
    img = Image.open(os.path.join(image_path)).convert("RGB")
    print(f'img size: {img.size} {box}')
    x1, y1, x2, y2 = box
    if (x2<x1) or (y2<y1):
        count += 1

print(f'dict: {dict.keys()} {count}')
