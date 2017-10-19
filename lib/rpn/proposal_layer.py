# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np
import numpy.random as npr
import json
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms
from utils.cython_bbox import bbox_overlaps

DEBUG = False

class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = json.loads(self.param_str)

        self._feat_stride = layer_params['feat_stride']
        self._has_cls = layer_params.get('has_cls', False)
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]

        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)
        if self.phase == caffe.TRAIN:
            # rois_labels
            top[1].reshape(1,)
            # rois_gt_assignments
            top[2].reshape(1,)


    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'
        if self.phase == caffe.TRAIN:
            assert len(bottom) == 4
            assert len(top) == 3
            assert bottom[3].shape[1] == 4
        else:
            assert len(bottom) == 3
            assert len(top) == 1
            assert bottom[2].shape[1] == 4

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[0].data[:, self._num_anchors:, :, :]
        bbox_deltas = bottom[1].data
        im_info = bottom[2].data[0, :]

        # generate RPN proposals
        proposals = self._generate_rpn_rois(scores, bbox_deltas, im_info)

        # sample proposals according to ground truth bboxes
        if self.phase == caffe.TRAIN:
            gt_boxes = bottom[3].data
            proposals, labels, gt_assignment = self._match_gt(proposals, gt_boxes)

        # image index is 0 for all proposals
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        proposals = np.hstack((batch_inds,
            proposals.astype(np.float32, copy=False)))
        top[0].reshape(*(proposals.shape))
        top[0].data[...] = proposals
        if self.phase == caffe.TRAIN:
            top[1].reshape(*(labels.shape))
            top[1].data[...] = labels
            top[2].reshape(*(gt_assignment.shape))
            top[2].data[...] = gt_assignment

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _generate_rpn_rois(self, scores, bbox_deltas, im_info):
        cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH

        if DEBUG:
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        if DEBUG:
            print 'score map size: {}'.format(scores.shape)

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove small predicted boxes (we removed this step)

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        assert len(keep) == post_nms_topN, \
                '{} vs {}'.format(len(keep), post_nms_topN)
        proposals = proposals[keep, :]
        return proposals

    def _compute_sample_num(self, n_pos, n_neg):
        batch_size = cfg.TRAIN.BATCH_SIZE
        # detection number
        n_sel_pos = min(n_pos, cfg.TRAIN.FG_FRACTION * batch_size)
        n_sel_neg = min(n_neg, batch_size - n_sel_pos)
        if self._has_cls:
            n_cls_pos = min(n_pos, batch_size)
            n_sel_pos = max(n_cls_pos, n_sel_pos)
        return int(n_sel_pos), int(n_sel_neg)

    def _match_gt(self, proposals, gt_boxes):
        assert proposals.shape[1] == 4
        assert gt_boxes.shape[1] == 4
        num_proposals = len(proposals)
        overlaps = bbox_overlaps(proposals.astype(np.float, copy=False),
                gt_boxes.astype(np.float, copy=False))
        max_overlaps = overlaps.max(axis=1)
        max_overlap_inds = overlaps.argmax(axis=1)
        pos_inds = np.where(max_overlaps > cfg.TRAIN.FG_THRESH)[0]
        neg_inds = np.where(np.logical_and(
            max_overlaps < cfg.TRAIN.BG_THRESH_HI,
            max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]

        # select samples
        n_pos = len(pos_inds)
        n_neg = len(neg_inds)
        n_sel_pos, n_sel_neg = self._compute_sample_num(n_pos, n_neg)
        n_keep = n_sel_pos + n_sel_neg

        if n_pos > 0:
            pos_inds = npr.choice(pos_inds, size=n_sel_pos, replace=False)
        if n_neg > 0:
            neg_inds = npr.choice(neg_inds, size=n_sel_neg, replace=False)
        keep_inds = np.append(pos_inds, neg_inds)
        assert len(keep_inds) == n_keep

        # assign labels
        labels = np.empty((n_keep,), dtype=np.float32)
        labels.fill(-1)
        labels[:n_sel_pos] = 1
        labels[n_sel_pos:] = 0

        proposals = proposals[keep_inds].astype(np.float32)
        gt_assignment = np.empty((n_keep,), dtype = np.float32)
        gt_assignment.fill(-1)
        gt_assignment[:n_sel_pos] = max_overlap_inds[pos_inds]

        return proposals, labels, gt_assignment
