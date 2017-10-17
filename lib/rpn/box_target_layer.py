import numpy as np
import numpy.random as npr
import caffe
from fast_rcnn.bbox_transform import bbox_transform
from fast_rcnn.config import cfg

class BoxTargetLayer(caffe.Layer):
    def setup(self, bottom, top):
        """
        bottoms:
            0: rois_labels
            1: rois_gt_assignment
            2: rois_boxes as src boxes
            3: gt_boxes as dst boxes

        tops:
            0: bbox_targets
            1: bbox_inside_weights
            2: bbox_outside_weights
        """

        assert len(bottom) == 4
        assert len(top) == 3
        top[0].reshape(1, 4)
        top[1].reshape(1, 4)
        top[2].reshape(1, 4)

    def forward(self, bottom, top):
        rois_labels = bottom[0].data
        gt_assignment = bottom[1].data.astype(np.int32)
        rois_boxes = bottom[2].data
        gt_boxes = bottom[3].data
        num_rois = len(rois_labels)
        assert len(gt_assignment) == num_rois, \
                "{} vs {}".format(len(gt_assignment), num_rois)
        assert len(rois_boxes) == num_rois, \
                "{} vs {}".format(len(rois_boxes), num_rois)
        assert rois_boxes.shape[1] == 5, rois_boxes.shape
        assert np.all(rois_boxes[:, 0] == 0), rois_boxes

        bbox_targets = np.zeros((num_rois, 4)).astype(np.float32)
        bbox_inside_weights = np.zeros((num_rois, 4)).astype(np.float32)
        bbox_outside_weights = np.zeros((num_rois, 4)).astype(np.float32)

        # sample rois
        pos_inds = np.where(rois_labels == 1)[0]
        n_pos = len(pos_inds)
        if n_pos:
            n_sel_pos = min(n_pos, cfg.TRAIN.BATCH_SIZE)
            fg_sel_inds = npr.choice(pos_inds, size=n_sel_pos, replace=False)
            src_boxes = rois_boxes[fg_sel_inds, 1:]
            dst_boxes = gt_boxes[gt_assignment[fg_sel_inds, 0]]
            targets = bbox_transform(src_boxes, dst_boxes)
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                    / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
            bbox_targets[fg_sel_inds] = targets
            bbox_inside_weights[fg_sel_inds] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
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
