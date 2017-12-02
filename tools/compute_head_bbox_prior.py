#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import numpy.random as npr
import logging
import cv2
import matplotlib.pyplot as plt
import random

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform, bbox_transform_inv
from fast_rcnn.train import get_training_roidb
from datasets.factory import get_imdb

np.set_printoptions(suppress=True)

def show_box(ax, boxes, head_boxes, color=None, labels=None):
    assert boxes.shape[1] == 4
    assert boxes.shape[0] == head_boxes.shape[0]
    for i in range(len(boxes)):
        c = color if color else npr.random(3)
        # body
        x1, y1, x2, y2 = boxes[i]
        coords = (x1, y1), x2-x1+1, y2-y1+1
        rect_kwargs = dict(fill=False, edgecolor=c, linewidth=2)
        ax.add_patch(plt.Rectangle(*coords, **rect_kwargs))
        # head
        h_x1, h_y1, h_x2, h_y2 = head_boxes[i]
        h_coords = (h_x1, h_y1), h_x2-h_x1+1, h_y2-h_y1+1
        rect_kwargs = dict(fill=True, edgecolor=c, linewidth=1, alpha=0.5)
        ax.add_patch(plt.Rectangle(*h_coords, **rect_kwargs))
        # label
        if labels is not None:
            text_kwargs = dict(size='small', color='white',
                    bbox=dict(facecolor=c, alpha=0.5, pad=0.15))
            ax.text(x1-2, y1-2, labels[i], **text_kwargs)
            ax.text(h_x1-2, h_y1-2, labels[i], **text_kwargs)

def where_valid(head_boxes, body_boxes):
    head_cents = np.vstack((
        (head_boxes[:, 0] + head_boxes[:, 2]) / 2,
        (head_boxes[:, 1] + head_boxes[:, 3]) / 2,
        )).T
    valid_inds = np.where(
            np.any(head_boxes != 0., axis=1)      &
            (head_cents[:, 0] > body_boxes[:, 0]) & # > x1
            (head_cents[:, 0] < body_boxes[:, 2]) & # < x2
            (head_cents[:, 1] > body_boxes[:, 1]) & # > y1
            (head_cents[:, 1] < body_boxes[:, 3])   # < y1
        )[0]
    return valid_inds

# [-0.         -0.23825483 -0.85038573 -0.68897909]

if __name__ == "__main__":
    ds_name = 'ftdata_train'
    imdb = get_imdb(ds_name)
    imdb.set_proposal_method('gt')
    roidb = get_training_roidb(imdb)

    # sample to compute prior transform
    n_all_samples = len(roidb)
    n_samp = 100
    samp_idx = random.sample(range(n_all_samples), min(n_samp, n_all_samples))
    print 'sampled {} images.'.format(len(samp_idx))
    body_boxes = np.vstack(roidb[i]['body_boxes'] for i in samp_idx)
    head_boxes = np.vstack(roidb[i]['head_boxes'] for i in samp_idx)
    print 'load {} boxes.'.format(len(body_boxes))

    # valid
    valid_inds = where_valid(head_boxes, body_boxes)
    body_boxes = body_boxes[valid_inds]
    head_boxes = head_boxes[valid_inds]

    # transform
    trans_params = bbox_transform(body_boxes, head_boxes)
    print '> trans_params:\n', trans_params
    trans_params = trans_params.mean(axis=0)
    print '> mean:\n', trans_params

    # show transformed example
    show_idx = 30
    body_boxes = roidb[show_idx]['body_boxes']
    head_trans_boxes = bbox_transform_inv(body_boxes,
        np.tile(trans_params, (len(body_boxes), 1)))

    image_path = imdb.image_path_at(show_idx)
    im = cv2.imread(image_path)[:, :, [2, 1, 0]]
    plt.imshow(im)
    ax = plt.gca()
    show_box(ax, body_boxes, head_trans_boxes)
    plt.title(image_path)
    plt.show()
