#!/usr/bin/env python3

import numpy as np
from PIL import Image
import skimage.draw

#from helpers.writers import BOX_COLOR_LUT

BOX_COLOR_LUT = np.array([(220, 20, 60), (255, 0, 0), ( 0, 0,142), ( 0, 0, 70), ( 0, 60,100), ( 0, 80,100), ( 0, 0,230),
                          (119, 11, 32), (250,170, 30), (220,220, 0), (220, 0, 220), (60, 220, 200), (0, 255, 0), (0, 150, 0)])

OUT_PATH = "../aachen_boxes.png"
image_orid = np.asarray(Image.open('../aachen.png'))
image = np.copy(image_orid)

inference_width = image.shape[1]
inference_height = image.shape[0]

colors = [BOX_COLOR_LUT[i,:] for i in range(BOX_COLOR_LUT.shape[0])]
print(colors)
boxes = []

class Box:
    def __init__(self, x, y, w, h, cls):
        self.x1 = x
        self.x2 = x+h
        self.y1 = y
        self.y2 = y+w
        self.cls = cls


for i, color in enumerate(colors):
    b = Box((i + 3) * 40, 40, 20, 20, i)
    boxes.append(b)

for box in boxes:
    cls = box.cls
    x1 = np.clip(box.x1, 0, inference_width - 1)
    y1 = np.clip(box.y1, 0, inference_height - 1)
    x2 = np.clip(box.x2, 0, inference_width - 1)
    y2 = np.clip(box.y2, 0, inference_height - 1)
    for coords in [[int(y1), int(x1), int(y1), int(x2)],
                   [int(y1), int(x2), int(y2), int(x2)],
                   [int(y2), int(x2), int(y2), int(x1)],
                   [int(y2), int(x1), int(y1), int(x1)]]:
        rr, cc = skimage.draw.line(*coords)
        image[rr, cc, :] = BOX_COLOR_LUT[cls]
        if coords[0] == coords[2]:
            if coords[0] > 0:
                image[rr - 1, cc, :] = BOX_COLOR_LUT[cls]
            if coords[0] < inference_height - 1:
                image[rr + 1, cc, :] = BOX_COLOR_LUT[cls]
        if coords[1] == coords[3]:
            if coords[1] > 0:
                image[rr, cc - 1, :] = BOX_COLOR_LUT[cls]
            if coords[1] < inference_width - 1:
                image[rr, cc + 1, :] = BOX_COLOR_LUT[cls]
image = Image.fromarray(image.astype(np.uint8), 'RGB')
image.save(OUT_PATH)