import numpy as np
import numpy.random as npr
import caffe
import json
from fast_rcnn.bbox_transform import bbox_transform
from fast_rcnn.config import cfg

def filter_valid(src_boxes, dst_boxes):
    """
    choose valid pairs:
    1. dst boxes not [0, 0, 0, 0]
    2. dst center in src boxes
    """
    cents = np.vstack((
        (dst_boxes[:, 0] + dst_boxes[:, 2]) / 2,
        (dst_boxes[:, 1] + dst_boxes[:, 3]) / 2,
        )).T
    valid_inds = np.where(
            np.any(dst_boxes != 0., axis=1) & # not all zero
            (cents[:, 0] > src_boxes[:, 0]) & # > x1
            (cents[:, 0] < src_boxes[:, 2]) & # < x2
            (cents[:, 1] > src_boxes[:, 1]) & # > y1
            (cents[:, 1] < src_boxes[:, 3])   # < y1
        )[0]
    return valid_inds


class BoxTargetLayer(caffe.Layer):
    def setup(self, bottom, top):
        """
        bottoms:
            0: rois_labels
            1: rois_gt_assignments
            2: rois_boxes as src boxes
            3: gt_boxes as dst boxes

        tops:
            0: bbox_targets
            1: bbox_inside_weights
            2: bbox_outside_weights
        """

        assert len(bottom) == 4
        assert len(top) == 3

        layer_params = {}
        if self.param_str:
            layer_params = json.loads(self.param_str)
        self.bbox_normalize_means = layer_params.get('bbox_normalize_means',
                cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        self.bbox_normalize_stds = layer_params.get('bbox_normalize_stds',
                cfg.TRAIN.BBOX_NORMALIZE_STDS)

        top[0].reshape(1, 4)
        top[1].reshape(1, 4)
        top[2].reshape(1, 4)

    def forward(self, bottom, top):
        rois_labels = bottom[0].data
        gt_assignments = bottom[1].data.astype(np.int32)
        rois_boxes = bottom[2].data
        gt_boxes = bottom[3].data

        num_rois = len(rois_labels)
        assert len(gt_assignments) == num_rois, \
                "{} vs {}".format(len(gt_assignments), num_rois)
        assert len(rois_boxes) == num_rois, \
                "{} vs {}".format(len(rois_boxes), num_rois)
        assert rois_boxes.shape[1] == 5, rois_boxes.shape
        # only support single image
        assert np.all(rois_boxes[:, 0] == 0), rois_boxes

        bbox_targets = np.zeros((num_rois, 4), dtype = np.float32)
        bbox_inside_weights = np.zeros((num_rois, 4), dtype = np.float32)
        bbox_outside_weights = np.zeros((num_rois, 4), dtype = np.float32)

        # sample rois
        pos_inds = np.where(rois_labels == 1)[0]
        n_pos = len(pos_inds)
        if n_pos > 0:
            # dst boxes
            dst_boxes = np.zeros((num_rois, 4), dtype=np.float32)
            dst_boxes[pos_inds] = gt_boxes[gt_assignments[pos_inds]]
            # choose valid boxes
            pos_sel_inds = filter_valid(rois_boxes, dst_boxes)
            # targets
            targets = bbox_transform(rois_boxes[pos_sel_inds, 1:],
                    dst_boxes[pos_sel_inds])
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                targets = ((targets - np.array(self.bbox_normalize_means))
                    / np.array(self.bbox_normalize_stds))
                # assert np.all(targets < 10), targets
            bbox_targets[pos_sel_inds] = targets
            bbox_inside_weights[pos_sel_inds] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
            bbox_outside_weights[bbox_inside_weights > 0] = 1.

        top[0].reshape(*(bbox_targets.shape))
        top[0].data[...] = bbox_targets
        top[1].reshape(*(bbox_inside_weights.shape))
        top[1].data[...] = bbox_inside_weights
        top[2].reshape(*(bbox_outside_weights.shape))
        top[2].data[...] = bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        pass
