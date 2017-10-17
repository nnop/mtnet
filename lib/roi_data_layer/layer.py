# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
import numpy as np
import numpy.random as npr
import json
import logging
import cv2

from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

class RoIDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()

    def setup(self, bottom, top):
        """Setup the RoIDataLayer.
        blobs:
            - data
            - im_info [hei, wid, scale]
            - gt_body [x1, y1, x2, y2, cls, stu]
            - gt_head [x1, y1, x2, y2, cls]
        """
        assert cfg.TRAIN.HAS_RPN, 'Only support PRN'

        # parse the layer parameter string, which must be valid YAML
        if self.param_str:
            layer_params = json.loads(self.param_str)
        self.has_head = False

        self._name_to_top_map = {'data': 0,
                              'im_info': 1,
                              'gt_body': 2}
        # data
        top[0].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
            max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
        # im_info
        top[1].reshape(1, 3)
        # body_boxes
        top[2].reshape(1, 6)
        if len(top) == 4:
            self.has_head = True
            # head_boxes
            top[3].reshape(1, 5)
            self._name_to_top_map['gt_head'] = 3

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.
        Construct a minibatch:
            - data (1, 3, H, W)
            - im_info (1, 3)
            - gt_body (N, 6) [x1, y1, x2, y2, cls, stu]
            - gt_head (N, 5) [x1, y1, x2, y2, cls]
        """
        assert cfg.TRAIN.HAS_RPN, 'Only support RPN.'
        db_inds = self._get_next_minibatch_inds()
        assert len(db_inds) == 1, 'Only support Single batch.'
        minibatch_db = [self._roidb[i] for i in db_inds]
        # Sample random scales to use for each image in this batch
        random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                        size=len(db_inds))
        # Get the input image blob, formatted for caffe
        im_blob, im_scales = _get_image_blob(minibatch_db, random_scale_inds)
        # blobs
        blobs = {}
        # data
        blobs['data'] = im_blob
        # im_info
        hei, wid, scale = im_blob.shape[2], im_blob.shape[3], im_scales[0]
        blobs['im_info'] = np.array([[hei, wid, scale]], dtype=np.float32)
        # body_boxes
        im_roi = minibatch_db[0]
        logging.info('Flipped: {}'.format(im_roi['flipped']))
        blobs['gt_body'] = np.hstack((
            im_roi['body_boxes'] * scale,
            im_roi['body_classes'][:, None],
            im_roi['stu_classes'][:, None]))
        if self.has_head:
            # head_boxes
            blobs['gt_head'] = np.hstack((
                im_roi['head_boxes'] * scale,
                im_roi['head_classes'][:, None]))
        return blobs

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        logging.info(roidb[i]['image'])
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales
