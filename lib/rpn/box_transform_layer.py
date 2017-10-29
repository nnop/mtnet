import caffe
import numpy as np

from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform_inv

class BoxTransformLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 2
        assert len(top) == 1
        boxes = bottom[0].data
        _dim = boxes.shape[1]
        assert _dim == 4 or _dim == 5
        n_boxes = bottom[0].data.shape[0]
        top[0].reshape(n_boxes, 5)

    def forward(self, bottom, top):
        """
        bottom:
            0: boxes
            1: deltas
        top:
            0: boxes_out
        """
        boxes = bottom[0].data
        n_boxes, n_dim = boxes.shape
        assert n_dim == 4 or n_dim == 5
        if n_dim == 5:
            boxes = boxes[:, 1:]
        deltas = bottom[1].data

        box_means = np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        box_stds = np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        denorm_deltas = deltas * box_stds + box_means
        boxes_out = bbox_transform_inv(boxes, denorm_deltas)
        boxes_out = np.hstack((np.zeros((n_boxes, 1)), boxes_out))

        # return
        top[0].reshape(*(boxes_out.shape))
        top[0].data[...] = boxes_out.astype(np.float32)

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        pass
