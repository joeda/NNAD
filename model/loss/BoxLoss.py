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

import tensorflow as tf
from .losses import *
from .WeightKendall import *

import numpy as np
import uuid
from pathlib import Path
import os

def img_summary(var, name, step):
    tf.print("var")
    tf.print(var)
    tf.print(tf.rank(var))
    if tf.rank(var) == 4:
        pass
        #tf.summary.image(name, var, step)
    else:
        tf.print("Variable " + name + " has rank ")

def ensure_dir(p):
    if os.path.exists(p) and not os.path.isdir(p):
        raise FileExistsError(p + " is not a directory and exists")
    Path(p).mkdir(exist_ok=True, parents=True)

class BoxLoss(tf.keras.Model):
    def __init__(self, name, config, box_delta_regression=False):
        super().__init__(name=name)

        self.num_bb_classes = config['num_bb_classes']
        self.box_embedding_len = config['box_embedding_len']

        self.weight_objectness = WeightKendall('weight_objectness')
        self.weight_class = WeightKendall('weight_class')
        self.weight_bb = WeightKendall('weight_bb')
        if box_delta_regression:
            self.weight_delta = WeightKendall('weight_delta')
        self.box_delta_regression = box_delta_regression

        self.cnt = 0

    def call(self, inputs, step, actual_step=None):
        prefix = str(uuid.uuid4().hex)
        write_dir = "/tmp/000_{}_{}".format(prefix, actual_step)
        ensure_dir(write_dir)
        results, ground_truth = inputs

        targets_bb = results['bb_targets_offset']
        targets_cls = results['bb_targets_cls']
        targets_obj = results['bb_targets_objectness']
        gt_targets_bb = ground_truth['bb_targets_offset']
        gt_targets_cls = ground_truth['bb_targets_cls']
        gt_targets_obj = ground_truth['bb_targets_objectness']
        #tf.print("shape")
        #tf.print(tf.shape(gt_targets_obj))
        #tf.print(tf.shape(gt_targets_cls))
        #tf.print(tf.shape(gt_targets_bb))

        #tf.print(gt_targets_bb, output_stream="file://" + os.path.join(write_dir, "gt_targets_bb"), summarize=-1)
        #tf.print(gt_targets_obj, output_stream="file://" + os.path.join(write_dir, "gt_targets_obj"), summarize=-1)
        #tf.print(gt_targets_cls, output_stream="file://" + os.path.join(write_dir, "gt_targets_cls"), summarize=-1)

        targets_bb = tf.reshape(targets_bb, [-1, 4])
        targets_cls = tf.reshape(targets_cls, [-1, self.num_bb_classes])
        targets_obj = tf.reshape(targets_obj, [-1, 2])
        gt_targets_bb = tf.reshape(gt_targets_bb, [-1, 4])
        gt_targets_cls = tf.reshape(gt_targets_cls, [-1])
        gt_targets_obj = tf.reshape(gt_targets_obj, [-1])
        #np.save(np.array(gt_targets_bb), "/tmp/aa_{:06d}/gt_targets_bb".format(self.cnt))
        #tf.io.write_file("/tmp/0000_{}/gt_targets_obj".format(prefix), "gt_targets_bb")
        #with open(os.path.join(write_dir, "gt_targets_bb"), "w") as f:
        #tf.print("printing to {}".format(os.path.join(write_dir, "gt_targets_bb")))
        #tf.print(gt_targets_bb, output_stream="file://" + os.path.join(write_dir, "gt_targets_bb"))
        #tf.print("gt targets bb nonzero")
        #tf.print(tf.math.count_nonzero(gt_targets_bb))
        #tf.print("gt targets obj nonzero")
        #tf.print(tf.math.count_nonzero(gt_targets_obj))

        #tf.print("gt targets bb")
        #tf.print(gt_targets_bb)
        #tf.print("gt targets obj")
        #tf.print(gt_targets_obj)
        #tf.print("gt targets cls")
        #tf.print(gt_targets_cls)

        shape = tf.shape(targets_bb)
        num_anchors = tf.cast(shape[0], tf.float32)

        # -1 is the ignore label for boxes
        mask_obj = tf.not_equal(gt_targets_obj, tf.constant([-1]))
        #tf.io.write_file("/tmp/aa_{:06d}/mask_obj".format(self.cnt), mask_obj)
        masked_targets_obj = tf.boolean_mask(targets_obj, mask_obj)
        masked_gt_targets_obj = tf.boolean_mask(gt_targets_obj, mask_obj)
        #tf.io.write_file("/tmp/aa_{:06d}/masked_gt_targets_obj".format(self.cnt), masked_gt_targets_obj)


        # We do not care for bounding box regression or classification of targets that do not correspond to an object
        mask_bb = tf.equal(gt_targets_obj, tf.constant([1]))
        masked_targets_bb = tf.boolean_mask(targets_bb, mask_bb)
        #tf.print("masked targets bb")
        #tf.print(masked_targets_bb)
        masked_gt_targets_bb = tf.boolean_mask(gt_targets_bb, mask_bb)
        #tf.print("masked gt targets bb")
        #tf.print(masked_gt_targets_bb)
        #tf.io.write_file("/tmp/aa_{:06d}/masked_gt_targets_bb".format(self.cnt), masked_gt_targets_bb)
        masked_targets_cls = tf.boolean_mask(targets_cls, mask_bb)
        masked_gt_targets_cls = tf.boolean_mask(gt_targets_cls, mask_bb)

        masked_gt_targets_cls = tf.stop_gradient(masked_gt_targets_cls)
        masked_gt_targets_obj = tf.stop_gradient(masked_gt_targets_obj)
        masked_gt_targets_bb = tf.stop_gradient(masked_gt_targets_bb)
        obj_loss = sparse_focal_loss(logits=masked_targets_obj, labels=masked_gt_targets_obj)
        cls_loss = sparse_focal_loss(logits=masked_targets_cls, labels=masked_gt_targets_cls)
        #cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=masked_targets_cls,
        #                                                          labels=masked_gt_targets_cls)
        bb_loss = smooth_l1_loss(logits=masked_targets_bb, labels=masked_gt_targets_bb, delta=0.1)
        #tf.print("obj loss")
        #tf.print(obj_loss)
        #tf.print("bb loss")
        #tf.print(bb_loss)
        obj_loss = tf.reduce_sum(obj_loss) / num_anchors
        cls_loss = tf.reduce_sum(cls_loss) / num_anchors
        bb_loss = tf.reduce_sum(bb_loss) / num_anchors

        tf.summary.scalar('bb_cls_loss_before_scale', cls_loss, step)
        tf.summary.scalar('bb_obj_loss_before_scale', obj_loss, step)
        tf.summary.scalar('bb_box_loss_before_scale', bb_loss, step)

        tf.summary.scalar('masked_gt_targets_cls', tf.size(masked_gt_targets_cls), step)
        tf.summary.scalar('masked_targets_cls', tf.size(masked_targets_cls), step)
        tf.summary.scalar('masked_gt_targets_bb', tf.size(masked_gt_targets_bb), step)
        tf.summary.scalar('masked_targets_bb', tf.size(masked_targets_bb), step)
        tf.summary.scalar('masked_gt_targets_obj', tf.size(masked_gt_targets_obj), step)
        tf.summary.scalar('masked_targets_obj', tf.size(masked_targets_obj), step)


        # Apply some sensible scaling before loss weighting
        obj_loss *= 5000.0
        cls_loss *= 10000.0
        bb_loss *= 20000.0

        obj_loss = self.weight_objectness(obj_loss, step)
        cls_loss = self.weight_class(cls_loss, step)
        bb_loss = self.weight_bb(bb_loss, step)

        tf.summary.scalar('bb_cls_loss', cls_loss, step)
        tf.summary.scalar('bb_obj_loss', obj_loss, step)
        tf.summary.scalar('bb_box_loss', bb_loss, step)

        #img_summary(gt_targets_obj, "gt_targets_obj", step)
        #tf.summary.image("gt_targets_obj", gt_targets_obj, step)
        #tf.summary.image("gt_targets_bb", gt_targets_bb, step)
        #tf.summary.image("gt_targets_cls", gt_targets_cls, step)
        #tf.summary.image("mask_obj", mask_obj, step)
        #tf.summary.image("masked_gt_targets_obj", masked_gt_targets_obj, step)

        losses = [cls_loss, obj_loss, bb_loss]

        if self.box_delta_regression:
            targets_delta = results['bb_targets_delta']
            gt_targets_delta = ground_truth['bb_targets_delta']
            gt_targets_delta_valid = ground_truth['bb_targets_delta_valid']

            targets_delta = tf.reshape(targets_delta, [-1, 4])
            targets_delta_pos, targets_delta_size = tf.split(targets_delta, num_or_size_splits=2, axis=-1)
            gt_targets_delta = tf.reshape(gt_targets_delta, [-1, 4])
            gt_targets_delta_pos, gt_targets_delta_size = tf.split(gt_targets_delta, num_or_size_splits=2, axis=-1)
            gt_targets_delta_valid = tf.reshape(gt_targets_delta_valid, [-1])

            mask_delta = tf.equal(gt_targets_delta_valid, tf.constant(1))
            masked_targets_delta_pos = tf.boolean_mask(targets_delta_pos, mask_delta)
            masked_targets_delta_size = tf.boolean_mask(targets_delta_size, mask_delta)
            masked_gt_targets_delta_pos = tf.boolean_mask(gt_targets_delta_pos, mask_delta)
            masked_gt_targets_delta_size = tf.boolean_mask(gt_targets_delta_size, mask_delta)
            delta_loss = smooth_l1_loss(logits=masked_targets_delta_pos, labels=masked_gt_targets_delta_pos, delta=10.0) + \
                         smooth_l1_loss(logits=masked_targets_delta_size, labels=masked_gt_targets_delta_size, delta=0.1)
            delta_loss = tf.reduce_sum(delta_loss) / num_anchors

            # Apply some sensible scaling before loss weighting
            delta_loss *= 10000.0

            delta_loss = self.weight_delta(delta_loss, step)
            tf.summary.scalar('bb_delta_loss', delta_loss, step)
            losses += [delta_loss]

        self.cnt += 1
        return losses
