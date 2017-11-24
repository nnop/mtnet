#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import sys
import caffe
import cv2
import argparse
import logging
import os.path as osp

sys.path.insert(0, 'lib')
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.test import _get_image_blob
from fast_rcnn.nms_wrapper import nms
from datasets.ftdata import ftdata
from utils.blob import im_list_to_blob
from utils.logger import config_logger

body_classes = ftdata._body_classes
head_classes = ftdata._head_classes

def show_box(ax, boxes, color=None, labels=None):
    assert boxes.shape[1] == 4
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        coords = (x1, y1), x2-x1+1, y2-y1+1
        c = color if color else npr.random(3)
        rect_kwargs = dict(fill=False, edgecolor=c, linewidth=2)
        ax.add_patch(plt.Rectangle(*coords, **rect_kwargs))
        if labels is not None:
            text_kwargs = dict(size='small', color='white',
                    bbox=dict(facecolor=c, alpha=0.5, pad=0.15))
            ax.text(x1-2, y1-2, labels[i], **text_kwargs)

if __name__ == "__main__":
    config_logger()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
            help='deploy prototxt file')
    parser.add_argument('--weights', type=str,
            help='model weights file')
    parser.add_argument('--gpu', default=0, type=int,
            help='gpu id, -1 for cpu')
    parser.add_argument('--conf-thresh', default=0.6,
            help='detection confidence threshold')
    parser.add_argument('--nms-thresh', default=0.5,
            help='detection nms threshold')
    parser.add_argument('--save', type=str,
            help='detection save path')
    parser.add_argument('image_path')
    args = parser.parse_args()

    proto_p = args.model
    weights_p = args.weights
    gpu = args.gpu
    im_p = args.image_path
    save_p = args.save
    det_conf_thershold = args.conf_thresh
    det_nms_threshold = args.nms_thresh

    assert osp.isfile(im_p), '{} not exists.'.format(im_p)
    cfg.HAS_RPN = True

    if gpu == -1:
        logging.info('use cpu mode')
        caffe.set_mode_cpu()
    else:
        logging.info('use gpu {}'.format(gpu))
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
        cfg.GPU_ID = gpu

    # create net
    net = caffe.Net(proto_p, weights_p, caffe.TEST)

    # forward
    im = cv2.imread(im_p)
    im_blobs, im_scales = _get_image_blob(im)
    info_blob = np.array(
            [[im_blobs.shape[2], im_blobs.shape[3], im_scales[0]]],
            dtype=np.float32)
    net.blobs['data'].reshape(*(im_blobs.shape))
    net.blobs['im_info'].reshape(*(info_blob.shape))
    out_blobs = net.forward(data=im_blobs, im_info=info_blob)

    # probs
    body_roi_probs = net.blobs['det_probs-body'].data[:, 1]
    body_inds = np.where(body_roi_probs > det_conf_thershold)[0]
    # boxes
    body_boxes = net.blobs['det_boxes-body'].data[body_inds, 1:] / im_scales[0]
    head_boxes = net.blobs['head_boxes-body'].data[:, 1:] / im_scales[0]

    # nms
    body_boxes = np.hstack(
            (body_roi_probs[body_inds, None], body_boxes)).astype(np.float32)
    keep = nms(body_boxes, det_nms_threshold)
    body_keep_inds = body_inds[keep]
    body_boxes = clip_boxes(body_boxes[keep, 1:], im.shape)
    head_boxes = clip_boxes(head_boxes[body_keep_inds], im.shape)
    # student labels
    stu_labels = net.blobs['stu_scores-body'].data[body_keep_inds, 1] > 0.5
    # body pose labels
    body_pose_scores = net.blobs['pose_scores-body'].data[body_keep_inds]
    body_pose_labels = np.argmax(body_pose_scores, axis=1)
    # head pose labels
    head_pose_scores = net.blobs['pose_scores-head'].data[body_keep_inds]
    head_pose_labels = np.argmax(head_pose_scores, axis=1)

    # show
    im = im[..., [2, 1, 0]]
    plt.figure(figsize=(12, 8))
    plt.imshow(im)
    ax = plt.gca()
    body_tags = []
    for stu, pose in zip(stu_labels, body_pose_labels):
        body_tags.append('{}-{}' \
                .format('stu' if stu else 'parent', body_classes[pose]))
    show_box(ax, body_boxes, color='g', labels=body_tags)
    head_tags = [head_classes[i] for i in head_pose_labels]
    show_box(ax, head_boxes, color='r', labels=head_tags)
    plt.savefig(save_p)
    logging.info('result save to: '+save_p)
