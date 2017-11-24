#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import os 
import os.path as osp
import cv2
import sys
import numpy as np
import json
import argparse
import caffe

import _init_paths
from datasets.factory import get_imdb
from fast_rcnn.test import _get_image_blob
from fast_rcnn.bbox_transform import clip_boxes
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.config import cfg
from utils.logger import config_logger
from datasets.ftdata import ftdata

body_classes = ftdata._body_classes
head_classes = ftdata._head_classes

np.set_printoptions(suppress=True)

def make_if_not_exists(d):
    if d and not osp.isdir(d):
        os.makedirs(d)

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
                    .format('stu' if stu else 'par', body_classes[pose]))
        show_box(ax, body_boxes, color='g', labels=body_tags)
        head_tags = [head_classes[i] for i in head_pose_labels]
        show_box(ax, head_boxes, color='r') #, labels=head_tags)
        out_p = vis_prefix+'_vis.png'
        plt.savefig(out_p)
        logging.info('Detection vis save to '+out_p)

def load_annotation(dump_p):
    with open(dump_p) as f:
        anno_info = json.load(f)
    n_person = len(anno_info['persons'])
    b_gt = False
    if 'meta' in anno_info:
        det_thresh = anno_info['meta']['conf_thresh']
        nms_thresh = anno_info['meta']['nms_thresh']
    else:
        b_gt = True

    # preallocate array
    body_scores = np.empty(n_person).astype(np.float32)
    body_boxes = np.empty((n_person, 4)).astype(np.float32)
    head_boxes = np.empty((n_person, 4)).astype(np.float32)
    body_pose_labels = np.empty(n_person).astype(np.int32)
    head_pose_labels = np.empty(n_person).astype(np.int32)
    stu_labels = np.empty(n_person).astype(np.bool)

    # fill array
    persons = anno_info['persons']
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

    # nms to filter out multiple parent boxes
    if b_gt:
        body_dets = np.hstack((np.zeros((n_person, 1)), body_boxes)).astype(np.float32)
        nms_keep = nms(body_dets, 0.6)
        body_scores = body_scores[nms_keep]
        body_boxes = body_boxes[nms_keep]
        head_boxes = head_boxes[nms_keep]
        body_pose_labels = body_pose_labels[nms_keep]
        head_pose_labels = head_pose_labels[nms_keep]
        stu_labels = stu_labels[nms_keep]

    if b_gt:
        logging.info('Load {} GTs from {}'.format(n_person, dump_p))
    else:
        logging.info('Load {} detections from {} @ ({}, {})' \
                .format(n_person, dump_p, det_thresh, nms_thresh))

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

def compute_iou(gt_boxes, det_box):
    assert gt_boxes.ndim == 1 or gt_boxes.ndim == 2, \
            'gt_boxes wrong shape: {}'.format(gt_boxes.shape)
    if gt_boxes.ndim == 1:
        gt_boxes = gt_boxes[None, :]
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

    # max IoU
    iou = inters / uni
    return iou

def voc_eval(all_dets, gt_anno_dir):
    # load gt annotations for all detections
    all_gts = {}
    npos = 0
    for im_name in all_dets:
        gt_anno_p = osp.join(gt_anno_dir, im_name+'.json')
        body_scores, body_boxes, head_boxes, stu_labels, \
            body_pose_labels, head_pose_labels = load_annotation(gt_anno_p)
        n_person = len(body_scores)
        all_gts[im_name] = {}
        all_gts[im_name]['body_scores'] = body_scores
        all_gts[im_name]['body_boxes'] = body_boxes
        all_gts[im_name]['head_boxes'] = head_boxes
        all_gts[im_name]['stu_labels'] = stu_labels
        all_gts[im_name]['body_pose_labels'] = body_pose_labels
        all_gts[im_name]['head_pose_labels'] = head_pose_labels
        all_gts[im_name]['assigned'] = [-1] * n_person
        npos += n_person

    # flatten detection results
    all_image_ids = []
    all_scores = []
    all_body_boxes = []
    all_head_boxes = []
    all_body_labels = []
    all_head_labels = []
    all_stu_labels = []
    for im_name in all_dets:
        info = all_dets[im_name]
        all_image_ids.extend([im_name] * len(info['body_scores']))
        all_scores.extend(info['body_scores'].tolist())
        all_body_boxes.extend(info['body_boxes'].tolist())
        all_head_boxes.extend(info['head_boxes'].tolist())
        all_body_labels.extend(info['body_pose_labels'].tolist())
        all_head_labels.extend(info['head_pose_labels'].tolist())
        all_stu_labels.extend(info['stu_labels'].tolist())
    all_scores = np.array(all_scores)
    all_body_boxes = np.array(all_body_boxes)
    all_head_boxes = np.array(all_head_boxes)
    all_body_labels = np.array(all_body_labels)
    all_head_labels = np.array(all_head_labels)
    all_stu_labels = np.array(all_stu_labels)

    # sort by confidence
    sorted_ind = np.argsort(-all_scores)
    all_image_ids = [all_image_ids[x] for x in sorted_ind]
    all_scores = all_scores[sorted_ind]
    all_body_boxes = all_body_boxes[sorted_ind]
    all_head_boxes = all_head_boxes[sorted_ind]
    all_body_labels = all_body_labels[sorted_ind]
    all_head_labels = all_head_labels[sorted_ind]
    all_stu_labels = all_stu_labels[sorted_ind]

    # mark TPs and FPs
    nd = len(all_image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        gt_info = all_gts[all_image_ids[d]]
        det_box = all_body_boxes[d]
        ovmax = -np.inf
        gt_boxes = gt_info['body_boxes']
        if gt_boxes.size > 0:
            iou = compute_iou(gt_boxes, det_box)
            ovmax = np.max(iou)
            jmax = np.argmax(iou)
        if ovmax > 0.5:
            if gt_info['assigned'][jmax] == -1:
                tp[d] = 1.
                # record the detection index
                gt_info['assigned'][jmax] = d
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute detection precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)
    logging.info('------ result ------')
    logging.info('detection AP on {} samples: {:.3f}'.format(len(all_dets.keys()), ap))

    # compute body pose confusion matrix
    num_body_class = len(body_classes)
    body_conf_mat = np.zeros((num_body_class, num_body_class))
    body_miss = 0
    body_all = 0
    for im_name in all_gts:
        gt_info = all_gts[im_name]
        assigned = gt_info['assigned']
        pose_labels = gt_info['body_pose_labels']
        n_person = len(assigned)
        body_all += n_person
        # loop against all objects in an image
        for i in range(n_person):
            gt_lab = pose_labels[i]
            if gt_lab == -1:   # no valid label
                continue
            d = assigned[i]
            if d == -1: # not detected
                body_miss += 1
                continue
            det_lab = all_body_labels[d]
            body_conf_mat[gt_lab, det_lab] += 1
    body_acc = np.sum(np.diag(body_conf_mat)) / np.sum(body_conf_mat)
    logging.info('body pose accuracy: {:.2f}, miss: {:.2f} ({}/{})\n{}\n{}' \
            .format(body_acc, float(body_miss) / body_all, body_miss, body_all,
                body_classes, body_conf_mat))

    # compute head pose confusion matrix
    num_head_class = len(head_classes)
    head_conf_mat = np.zeros((num_head_class, num_head_class))
    head_miss = 0
    head_all = 0
    for im_name in all_gts:
        gt_info = all_gts[im_name]
        assigned = gt_info['assigned']
        pose_labels = gt_info['head_pose_labels']
        gt_boxes = gt_info['head_boxes']
        n_person = len(assigned)
        # loop against all objects in an image
        for i in range(n_person):
            gt_lab = pose_labels[i]
            gt_box = gt_boxes[i]
            if np.all(gt_box == 0.):    # no valid head
                continue
            head_all += 1
            if gt_lab == -1:            # no valid label
                continue
            d = assigned[i]
            if d == -1: # not detected
                head_miss += 1
                continue
            det_lab = all_head_labels[d]
            det_box = all_head_boxes[d]
            iou = compute_iou(gt_box, det_box)
            if iou[0] < 0.5:
                head_miss += 1
                continue
            head_conf_mat[gt_lab, det_lab] += 1
    head_acc = np.sum(np.diag(head_conf_mat)) / np.sum(head_conf_mat)
    logging.info('head pose accuracy: {:.2f}, miss: {:.2f} ({}/{})\n{}\n{}' \
            .format(head_acc, float(head_miss) / head_all, head_miss, head_all,
                head_classes, head_conf_mat))

    # compute student/parent confusion matrix
    stu_conf_mat = np.zeros((2, 2))
    for im_name in all_gts:
        gt_info = all_gts[im_name]
        assigned = gt_info['assigned']
        stu_labels = gt_info['stu_labels']
        n_person = len(assigned)
        # loop against all objects in an image
        for i in range(n_person):
            gt_lab = int(stu_labels[i])
            d = assigned[i]
            det_lab = int(all_stu_labels[d])
            if gt_lab == det_lab:
                stu_conf_mat[gt_lab, det_lab] += 1
    stu_acc = np.sum(np.diag(stu_conf_mat)) / np.sum(stu_conf_mat)
    logging.info('student accuracy: {:.2f}\n{}\n{}' \
            .format(stu_acc, ['parent', 'student'], stu_conf_mat))

    return rec, prec, ap, body_conf_mat, head_conf_mat, stu_conf_mat

def test_net(net, imdb, det_dir, gt_anno_dir, vis_dir='./',
        det_thresh=0.05, nms_thresh=0.5, vis=False):
    # collect all detections
    num_images = len(imdb.image_index)
    all_dets = {}
    for i in xrange(num_images):
        im_p = imdb.image_path_at(i)
        im_name = imdb.image_index[i]
        dump_p = osp.join(det_dir, im_name+'.json')
        if osp.isfile(dump_p) and net is None:
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
            vis_detections(im_p, det_res, osp.join(vis_dir, im_name))
        logging.info('Processed {}/{}: {} detections\n{}' \
                .format(i+1, num_images, n_person, im_p))

    # evaluation and save
    rec, prec, det_ap, body_conf_mat, head_conf_mat, stu_conf_mat \
            = voc_eval(all_dets, gt_anno_dir)
    plt.plot(rec, prec)
    plt.title('detection precision/recall curve')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plot_save_p = osp.join(det_dir, 'prec_rec.pdf')
    plt.savefig(plot_save_p)
    logging.info('precision/recall curve save to '+plot_save_p)
    # save body confusion matrix
    body_confmat_save_p = osp.join(det_dir, 'body_conf_mat.npy')
    np.save(body_confmat_save_p, body_conf_mat)
    logging.info('body pose confusion matrix save to '+body_confmat_save_p)
    # save head confusion matrix
    head_confmat_save_p = osp.join(det_dir, 'head_conf_mat.npy')
    np.save(head_confmat_save_p, head_conf_mat)
    logging.info('head pose confusion matrix save to '+head_confmat_save_p)
    # save stu confusion matrix
    stu_confmat_save_p = osp.join(det_dir, 'stu_conf_mat.npy')
    np.save(stu_confmat_save_p, stu_conf_mat)
    logging.info('student/parent confusion matrix save to '+stu_confmat_save_p)

if __name__ == "__main__":
    config_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
            help='deploy prototxt file')
    parser.add_argument('--weights', type=str,
            help='model weights file')
    parser.add_argument('--imdb', required=True,
            help='test imdb name')
    parser.add_argument('--gt-dir', required=True, type=str,
            help='ground truth label dir')
    parser.add_argument('--gpu', default=0, type=int,
            help='gpu id, -1 for cpu')
    parser.add_argument('--redo', action='store_true',
            help='redo evaluation by loading detection results from disk')
    parser.add_argument('--vis', action='store_true',
            help='visualize detection result')
    parser.add_argument('--vis-dir', default='./',
            help='visualization save dir')
    parser.add_argument('--det-dir', default='./output/',
            help='detection result save dir')

    args = parser.parse_args()
    imdb_name = args.imdb
    model_proto = args.model
    weights_path = args.weights
    gt_dir = args.gt_dir
    gpu = args.gpu
    redo = args.redo
    vis = args.vis
    det_dir = args.det_dir
    vis_dir = args.vis_dir

    # config
    cfg.TEST.HAS_RPN = True

    if gpu == -1:
        logging.info('use cpu mode')
        caffe.set_mode_cpu()
    else:
        logging.info('use gpu {}'.format(gpu))
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
        cfg.GPU_ID = gpu

    if redo:
        net = None 
    else:
        net = caffe.Net(model_proto, weights_path, caffe.TEST)

    make_if_not_exists(vis_dir)
    make_if_not_exists(det_dir)

    imdb = get_imdb(imdb_name)
    imdb.set_proposal_method('gt')
    test_net(net, imdb, det_dir=det_dir, gt_anno_dir=gt_dir,
            vis_dir=vis_dir, vis=vis)
