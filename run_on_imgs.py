#!/usr/bin/env python3

##########################################################################
# NNAD (Neural Networks for Automated Driving) training scripts          #
# Copyright (C) 2019 FZI Research Center for Information Technology      #
#                                                                        #
# This program is free software: you can redistribute it and/or modify   #
# it under the terms of the GNU General Public License as published by   #
# the Free Software Foundation, either version 3 of the License, or      #
# (at your option) any later version.                                    #
#                                                                        #
# This program is distributed in the hope that it will be useful,        #
# but WITHOUT ANY WARRANTY; without even the implied warranty of         #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          #
# GNU General Public License for more details.                           #
#                                                                        #
# You should have received a copy of the GNU General Public License      #
# along with this program.  If not, see <https://www.gnu.org/licenses/>. #
##########################################################################

import sys
sys.path.append('.')
import os
import time

import tensorflow as tf

from data import *
from helpers.configreader import *
from helpers.helpers import *
from helpers.writers import *
from evaluation.boundingboxevaluator import *
import argparse
import glob
from PIL import Image
from pathlib import Path
from progressbar import progressbar

BOX_COLOR_LUT = np.array([(220, 20, 60), (255, 0, 0), ( 0, 0,142), ( 0, 0, 70), ( 0, 60,100), ( 0, 80,100), ( 0, 0,230),
                          (119, 11, 32), (250,170, 30), (220,220, 0), (220, 0, 220), (60, 220, 200), (0, 255, 0), (0, 150, 0), (0,0,0)])

def get_imgs(path):
    return sorted([p for p in Path(path).rglob("*frankfurt*.png")])

def convert_image(image):
    image = image.numpy()[0, :, :, :]
    image = image * 255.0
    image = np.flip(image, -1) # BGR to RGB
    return image

def write_bbox_img(boxes, image, path, sz):
    image = convert_image(image)
    ensure_path(path)
    inference_height = np.shape(image)[0]
    inference_width = np.shape(image)[1]
    width, height = sz

    for box in boxes:
        cls = box.box.cls
        x1 = np.clip(box.box.x1, 0, inference_width - 1)
        y1 = np.clip(box.box.y1, 0, inference_height - 1)
        x2 = np.clip(box.box.x2, 0, inference_width - 1)
        y2 = np.clip(box.box.y2, 0, inference_height - 1)
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
    image = PIL.Image.fromarray(image.astype(np.uint8), 'RGB')
    image = image.resize((width, height), PIL.Image.BILINEAR)
    image.save(path)

def load_img(path, resize=(1024, 512)):
    with Image.open(path.absolute()) as f:
        f = f.resize(resize)
        img = np.array(f).astype(np.float32)
        img = np.flip(img, -1)
        img /= 255
        img = np.expand_dims(img, axis=0)
        return img

def my_fun(current_img=None):
    return 1

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("-i", type=str, required=True)
parser.add_argument("-o", type=str, required=True)
args = parser.parse_args()

config_path = args.config
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
bbutils = BBUtils(config['eval_image_width'], config['eval_image_height'])
# Load the saved model
saved_model_dir = os.path.join(config['state_dir'], 'saved_model')
model = tf.saved_model.load(saved_model_dir)
infer_fn = model.signatures['infer']
#infer_fn = my_fun

for relp in progressbar(get_imgs(args.i)):
    if relp.name.find("munster_000089_000016_leftImg8bit") != -1: #file is broken
        continue
    img = tf.convert_to_tensor(load_img(relp), dtype=tf.float32)
    sz = img.shape[2], img.shape[1]
    output = infer_fn(current_img=img)


    # if config['train_labels']:
    #     label_output = output['pixelwise_labels']
    #
    #     write_label_img(label_output, metadata, out_dir)
    #     write_debug_label_img(label_output, inp['left'], metadata, out_dir)

    if config['train_boundingboxes']:
        bb_targets_offset = output['bb_targets_offset'].numpy()
        bb_targets_cls = output['bb_targets_cls'].numpy()
        bb_targets_objectness = output['bb_targets_objectness'].numpy()
        bb_targets_embedding = output['bb_targets_embedding'].numpy()

        boxes = bbutils.bbListFromTargetsBuffer(bb_targets_objectness, bb_targets_cls, bb_targets_offset,
                                                np.array([]), bb_targets_embedding, 0.5)
        opath = os.path.join(args.o, relp.name)
        write_bbox_img(boxes, img, opath , sz)
