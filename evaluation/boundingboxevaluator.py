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
sys.path.append('..')

import numpy as np
import os
import csv

class BoundingBoxEvaluator(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.fp = [0 for i in range(num_classes)]
        self.tp = [0 for i in range(num_classes)]
        self.fn = [0 for i in range(num_classes)]
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.uint64)
        self.width_cnt = np.zeros(2000, dtype=np.uint64)

    def _iou(self, box1, boxes2):
        b1_x1, b1_y1, b1_x2, b1_y2 = np.split(box1, 4, axis=0)
        b2_x1, b2_y1, b2_x2, b2_y2 = np.split(boxes2, 4, axis=1)
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        assert np.all(b1_area >= 0)
        assert np.all(b2_area > 0)

        x_1 = np.maximum(b1_x1, b2_x1)
        y_1 = np.maximum(b1_y1, b2_y1)
        x_2 = np.minimum(b1_x2, b2_x2)
        y_2 = np.minimum(b1_y2, b2_y2)
        inter_area = np.maximum((x_2 - x_1 + 1), 0) * np.maximum((y_2 - y_1 + 1), 0)
        union_area = b1_area + b2_area - inter_area

        iou = np.divide(inter_area, union_area)
        assert np.all(np.isfinite(iou))
        return iou

    def add_confusion(self, gt_boxes, detection_list):
        min_width = 10
        min_height = 10
        max_height = 100
        max_width = 100

        gt_boxes = gt_boxes[0, :, :]

        detections = []
        for detection in detection_list:
            width = detection.box.x2 - detection.box.x1
            height = detection.box.y2 - detection.box.y1
            if width > min_width and height > min_height:
                detections += [[detection.box.x1, detection.box.y1, detection.box.x2, detection.box.y2]]
        detections = np.array(detections)
        gt_boxes_cls = gt_boxes[:, 0]
        gt_boxes_w = gt_boxes[:, 3] - gt_boxes[:, 1]
        gt_boxes_h = gt_boxes[:, 4] - gt_boxes[:, 2]
        masked_gt_boxes = gt_boxes[np.logical_and(gt_boxes_w > min_width, gt_boxes_h > min_height)]
        #masked_gt_boxes = masked_gt_boxes[:, 1:]
        if masked_gt_boxes[:, 1:].size == 0 or len(detections) == 0:
            return

        ious = np.array([self._iou(d, masked_gt_boxes[:, 1:]) for d in detections])
        for cnt in range(len(detections)):
            if ious.size == 0:
                break
            am = np.unravel_index(np.argmax(ious, axis=None), ious.shape)
            if ious[am] <= 0.5:
                break
            det_cls = detection_list[am[0]].box.cls
            gt_cls = masked_gt_boxes[am[1], 0]
            np.delete(ious, am[0], 0)
            np.delete(ious, am[1], 1)
            self.confusion[gt_cls, det_cls] += 1

    def add(self, gt_boxes, detection_list):
        # Ignore boxes that are too small
        min_width = 10
        min_height = 10
        max_height = 1000
        max_width = 60

        gt_boxes = gt_boxes[0, :, :]

        def add_cls(cls):
            detections = []
            for detection in detection_list:
                width = detection.box.x2 - detection.box.x1;
                height = detection.box.y2 - detection.box.y1
                if cls < 10:
                    if detection.box.cls == cls and width > min_width and height > min_height:
                        detections += [[detection.box.x1, detection.box.y1, detection.box.x2, detection.box.y2]]
                else:
                    if detection.box.cls > 9 and width > min_width and height > min_height and height < max_height and width < max_width:
                        detections += [[detection.box.x1, detection.box.y1, detection.box.x2, detection.box.y2]]
            detections = np.array(detections)
            gt_boxes_cls = gt_boxes[:, 0]
            gt_boxes_w = gt_boxes[:, 3] - gt_boxes[:, 1]
            gt_boxes_h = gt_boxes[:, 4] - gt_boxes[:, 2]
            if cls < 10:
                masked_gt_boxes = gt_boxes[np.logical_and(gt_boxes_cls == cls,
                                                          np.logical_and(gt_boxes_w > min_width, gt_boxes_h > min_height))]
            else:
                masked_gt_boxes = gt_boxes[np.logical_and(gt_boxes_cls > 9,
                                                          np.logical_and(gt_boxes_w > min_width,
                                                                         gt_boxes_h > min_height))]
            masked_gt_boxes = masked_gt_boxes[:, 1:]

            if np.shape(masked_gt_boxes)[0] == 0:
                self.fp[cls] += np.shape(detections)[0]
                return

            if np.shape(detections)[0] == 0:
                self.fn[cls] += np.shape(masked_gt_boxes)[0]
                return

            unused_box_idx = np.array([i for i in range(np.shape(masked_gt_boxes)[0])], dtype=np.int64)
            for i in range(np.shape(detections)[0]):
                detection = detections[i, :]
                if np.shape(unused_box_idx)[0] == 0:
                    self.fp[cls] += 1
                    continue
                bb_iou = self._iou(detection, masked_gt_boxes[unused_box_idx])
                bb_iou_idx = np.argmax(bb_iou)
                if bb_iou[bb_iou_idx] > 0.5:
                    self.tp[cls] += 1
                    unused_box_idx = np.delete(unused_box_idx, bb_iou_idx)
                else:
                    # TODO handle ignore areas
                    self.fp[cls] += 1
            self.fn[cls] += np.shape(unused_box_idx)[0]

        for cls in range(11):
            add_cls(cls)

    def print_results(self):
        print('Bounding box evaluation results:')
        for cls in range(self.num_classes):
            precision = self.tp[cls] / (self.tp[cls] + self.fp[cls]) if (self.tp[cls] + self.fp[cls]) > 0 else -1.0
            recall = self.tp[cls] / (self.tp[cls] + self.fn[cls]) if (self.tp[cls] + self.fn[cls]) > 0 else -1.0
            print('Class %d - Precision@0.5IoU: %f, Recall@0.5IoU: %f' % (cls, precision, recall))

    def dump(self, path):
        with open(os.path.join(path, "fp.txt"), "w") as f:
            fieldnames = ['fp', 'fn', "tp"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(11):
                writer.writerow({"fp": self.fp[i], "fn": self.fn[i], "tp": self.tp[i]})
        np.save(os.path.join(path, "confusion.npy"), self.confusion)
        np.save(os.path.join(path, "width.npy"), self.width_cnt)
