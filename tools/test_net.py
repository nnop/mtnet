#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os.path as osp
import cv2
import sys
import caffe
import numpy as np
import matplotlib.pyplot as plt
import json

import _init_paths
from datasets.factory import get_imdb
from fast_rcnn.test import _get_image_blob
from fast_rcnn.bbox_transform import clip_boxes
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.config import cfg
from utils.logger import config_logger
from datasets.ftdata import ftdata

from IPython import embed

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
            ax.text(x1-2, y1-2, labels[i][:2], **text_kwargs)


def im_detect(net, im, det_thresh, nms_thresh):
    im_blobs, im_scales = _get_image_blob(im)
    info_blob = np.array(
            [[im_blobs.shape[2], im_blobs.shape[3], im_scales[0]]],
            dtype=np.float32)
    net.blobs['data'].reshape(*(im_blobs.shape))
    net.blobs['im_info'].reshape(*(info_blob.shape))
    blobs_out = net.forward(data=im_blobs, im_info=info_blob)
    # thresholding detections
    body_roi_probs = net.blobs['det_probs-body'].data[:, 1]
    thresh_inds = np.where(body_roi_probs > det_thresh)[0]
    body_boxes = net.blobs['det_boxes-body'].data[:, 1:] / im_scales[0]
    body_boxes = clip_boxes(body_boxes, im.shape)
    body_dets = np.hstack((body_roi_probs[thresh_inds, None],
        body_boxes[thresh_inds])).astype(np.float32)
    # nms
    nms_keep = nms(body_dets, nms_thresh)
    keep_inds = thresh_inds[nms_keep]
    # body boxes
    body_scores = body_roi_probs[keep_inds]
    body_boxes = body_boxes[keep_inds]
    # head boxes
    head_boxes = net.blobs['head_boxes-body'].data[keep_inds, 1:] / im_scales[0]
    head_boxes = clip_boxes(head_boxes, im.shape)
    # student labels
    stu_labels = net.blobs['stu_scores-body'].data[keep_inds, 1] > 0.5
    # body pose labels
    body_pose_labels = np.argmax(net.blobs['pose_scores-body'].data[keep_inds],
            axis=1)
    # head pose labels
    head_pose_labels = np.argmax(net.blobs['pose_scores-head'].data[keep_inds],
            axis=1)
    return body_scores, body_boxes, head_boxes, stu_labels, \
           body_pose_labels, head_pose_labels

def vis_detections(im_p, det_res, vis_prefix):
        body_scores, body_boxes, head_boxes, stu_labels, body_pose_labels, \
                head_pose_labels = det_res
        im = cv2.imread(im_p)
        im = im[..., (2, 1, 0)]
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
        out_p = vis_prefix+'_vis.png'
        plt.savefig(out_p)
        logging.info('Detection vis save to '+out_p)

def load_annotation(dump_p):
    with open(dump_p) as f:
        det_info = json.load(f)
    n_person = len(det_info['persons'])
    if 'meta' in det_info:
        det_thresh = det_info['meta']['conf_thresh']
        nms_thresh = det_info['meta']['nms_thresh']
        logging.info('Loads {} detections from {} @ ({}, {})' \
                .format(n_person, dump_p, det_thresh, nms_thresh))
    else:
        logging.info('Loads {} bodies from {}'.format(n_person, dump_p))

    persons = det_info['persons']
    body_scores = np.empty(n_person).astype(np.float32)
    body_boxes = np.empty((n_person, 4)).astype(np.float32)
    head_boxes = np.empty((n_person, 4)).astype(np.float32)
    body_pose_labels = np.empty(n_person).astype(np.int32)
    head_pose_labels = np.empty(n_person).astype(np.int32)
    stu_labels = np.empty(n_person).astype(np.bool)
    for i in range(n_person):
        # body
        body_scores[i] = persons[i]['body'].get('score', 0.)
        body_boxes[i] = persons[i]['body']['bbox']
        b_lab = persons[i]['body']['label']
        if b_lab:
            body_pose_labels[i] = body_classes.index(b_lab)
        else:
            body_pose_labels[i] = -1
        stu_labels[i] = persons[i]['student']
        # head
        head_info = persons[i]['head']
        if head_info is None:
            head_boxes[i] = [0, 0, 0, 0]
            head_pose_labels[i] = -1
        else:
            head_boxes[i] = head_info['bbox']
            head_pose_labels[i] = head_classes.index(head_info['label'])
    return body_scores, body_boxes, head_boxes, stu_labels, \
            body_pose_labels, head_pose_labels

def save_detection(det_res, dump_p, det_thresh, nms_thresh):
    body_scores, body_boxes, head_boxes, stu_labels, \
        body_pose_labels, head_pose_labels = det_res
    n_person = len(body_scores)
    # dump
    det_info = {
        'meta': { 'conf_thresh': det_thresh, 'nms_thresh': nms_thresh },
        'persons': [] }
    for i in range(n_person):
        det_info['persons'].append(dict(
            body=dict(bbox=body_boxes[i].tolist(),
                label=body_classes[body_pose_labels[i]],
                score=body_scores[i].item()),
            head=dict(bbox=head_boxes[i].tolist(),
                label=head_classes[head_pose_labels[i]]),
            student=stu_labels[i].item()
        ))
    with open(dump_p, 'w') as f:
        json.dump(det_info, f)
    logging.info('Dump {} detections to {} @ ({}, {})' \
            .format(n_person, dump_p, det_thresh, nms_thresh))

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(all_dets, gt_anno_dir):
    # load gt annotations for all detections
    all_gts = {}
    npos = 0
    for im_name in all_dets:
        gt_anno_p = osp.join(gt_anno_dir, im_name+'.json')
        gt_info = load_annotation(gt_anno_p)
        body_scores, body_boxes, head_boxes, stu_labels, \
            body_pose_labels, head_pose_labels = gt_info
        n_person = len(body_scores)
        all_gts[im_name] = {}
        all_gts[im_name]['body_scores'] = body_scores
        all_gts[im_name]['body_boxes'] = body_boxes
        all_gts[im_name]['head_boxes'] = head_boxes
        all_gts[im_name]['stu_labels'] = stu_labels
        all_gts[im_name]['body_pose_labels'] = body_pose_labels
        all_gts[im_name]['head_pose_labels'] = head_pose_labels
        all_gts[im_name]['assigned'] = [False] * n_person
        npos += n_person

    all_image_ids = []
    all_scores = []
    all_body_boxes = []
    for im_name in all_dets:
        info = all_dets[im_name]
        all_image_ids.extend([im_name] * len(info['body_scores']))
        all_scores.extend(info['body_scores'].tolist())
        all_body_boxes.extend(info['body_boxes'].tolist())
    all_scores = np.array(all_scores)
    all_body_boxes = np.array(all_body_boxes)

    # sort by confidence
    sorted_ind = np.argsort(-all_scores)
    all_image_ids = [all_image_ids[x] for x in sorted_ind]
    all_scores = all_scores[sorted_ind]
    all_body_boxes = all_body_boxes[sorted_ind]

    # go down from dets and mark TPs and FPs
    nd = len(all_image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        gt_info = all_gts[all_image_ids[d]]
        det_box = all_body_boxes[d]
        ovmax = -np.inf
        gt_boxes = gt_info['body_boxes']
        if gt_boxes.size > 0:
            # intersection
            ixmin = np.maximum(gt_boxes[:, 0], det_box[0])
            iymin = np.maximum(gt_boxes[:, 1], det_box[1])
            ixmax = np.minimum(gt_boxes[:, 2], det_box[2])
            iymax = np.minimum(gt_boxes[:, 3], det_box[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
            # union
            uni = ((det_box[2] - det_box[0] + 1.) * (det_box[3] - det_box[1] + 1.) +
                   (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.) *
                   (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
        if ovmax > 0.5:
            if not gt_info['assigned'][jmax]:
                tp[d] = 1.
                gt_info['assigned'][jmax] = True
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric=False)
    return rec, prec, ap


def test_net(net, imdb, det_thresh=0.05, nms_thresh=0.3, det_dump_dir='./', vis=False):
    num_images = len(imdb.image_index)
    all_dets = {}
    for i in xrange(num_images):
        im_p = imdb.image_path_at(i)
        im_name = imdb.image_index[i]
        dump_p = osp.join(det_dump_dir, im_name+'.json')
        if osp.isfile(dump_p):
            # load from dump file
            det_res = load_annotation(dump_p)
        else:
            # perform detection
            im = cv2.imread(im_p)
            det_res = im_detect(net, im, det_thresh, nms_thresh)
            save_detection(det_res, dump_p, det_thresh, nms_thresh)
        body_scores, body_boxes, head_boxes, stu_labels, \
            body_pose_labels, head_pose_labels = det_res
        n_person = len(body_scores)
        all_dets[im_name] = {}
        all_dets[im_name]['body_scores'] = body_scores
        all_dets[im_name]['body_boxes'] = body_boxes
        all_dets[im_name]['head_boxes'] = head_boxes
        all_dets[im_name]['stu_labels'] = stu_labels
        all_dets[im_name]['body_pose_labels'] = body_pose_labels
        all_dets[im_name]['head_pose_labels'] = head_pose_labels
        # show
        if vis:
            vis_detections(im_p, det_res, im_name)
        logging.info('{}/{} {} @ {}'.format(i, num_images, im_p, n_person))
    # evaluation
    gt_anno_dir = './data/ftdata/Annotations/'
    rec, prec, ap = voc_eval(all_dets, gt_anno_dir)
    plt.plot(rec, prec)
    plt.savefig('pr.png')
    logging.info('AP: {:.3f}'.format(ap))

if __name__ == "__main__":
    config_logger()

    # config
    cfg.TEST.HAS_RPN = True

    imdb = get_imdb('ftdata_test')
    imdb.set_proposal_method('gt')
    caffe.set_mode_cpu()
    net = caffe.Net('models/roialign/deploy.prototxt',
            'output/roialign_iter_50000.caffemodel.h5', caffe.TEST)
    test_net(net, imdb, nms_thresh=0.5, det_dump_dir='./output/det_dump')
