import caffe
import numpy as np
import numpy.random as npr
import logging
import math
import ipdb

from fast_rcnn.config import cfg

DEBUG = False

class LabelTargetLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) == 1:
            self._use_neg = True
        elif len(bottom) == 3:
            self._use_neg = False
        else:
            raise ValueError('Wrong number of bottoms: {}'.format(len(bottom)))

        # labels
        top[0].reshape(1, 1)

    def forward(self, bottom, top):
        # tops
        #   
        rois_labels = bottom[0].data.astype(np.int32)
        if not self._use_neg:
            extra_labels = bottom[1].data.astype(np.int32)
            gt_assignment = bottom[2].data.astype(np.int32)

        # sample rois
        pos_inds = np.where(rois_labels == 1)[0]
        neg_inds = np.where(rois_labels == 0)[0]
        n_pos = len(pos_inds)
        n_neg = len(neg_inds)
        batch_size = cfg.TRAIN.BATCH_SIZE
        n_pos_per_image  = int(batch_size * cfg.TRAIN.FG_FRACTION)
        if self._use_neg:
            # use both pos and neg samples
            n_sel_pos = int(min(n_pos, n_pos_per_image))
            n_sel_neg = int(min(batch_size - n_sel_pos, n_neg))
        else:
            # use only pos samples
            n_sel_pos = int(min(n_pos, batch_size))
            n_sel_neg = 0

        if DEBUG:
            logging.info('---------')
            logging.info('(pos, neg / rois): ({}, {} / {})' \
                .format(n_pos, n_neg, len(rois_labels)))
            logging.info('batch_size: {}'.format(batch_size))
            logging.info('pos_per_image: {}'.format(n_pos_per_image))
            logging.info('select (pos, neg): ({}, {})' \
                .format(n_sel_pos, n_sel_neg))

        out_labels = np.empty((rois_labels.shape[0], 1), dtype=np.float32)
        out_labels.fill(-1)
        if n_sel_pos > 0:
            pos_sel_inds = npr.choice(pos_inds, size=n_sel_pos, replace=False)
            if self._use_neg:
                # fg/bg
                out_labels[pos_sel_inds, 0] = rois_labels[pos_sel_inds, 0]
            else:
                # use labels
                out_labels[pos_sel_inds, 0] = \
                    extra_labels[gt_assignment[pos_sel_inds, 0], 0]
        if n_sel_neg > 0:
            neg_sel_inds = npr.choice(neg_inds, size=n_sel_neg, replace=False)
            out_labels[neg_sel_inds, 0] = rois_labels[neg_sel_inds, 0]

        top[0].reshape(*(out_labels.shape))
        top[0].data[...] = out_labels

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        pass
