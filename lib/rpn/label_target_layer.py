import caffe
import numpy as np
import numpy.random as npr

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
        rois_labels = bottom[0].data
        if not self._use_neg:
            extra_labels = bottom[1].data
            gt_assignment = bottom[2].data.astype(np.int32)

        # sample rois
        pos_inds = np.where(rois_labels == 1)[0]
        neg_inds = np.where(rois_labels == 0)[0]
        n_pos = len(pos_inds)
        n_neg = len(neg_inds)
        batch_size = cfg.TRAIN.BATCH_SIZE
        n_pos_per_image  = batch_size * cfg.TRAIN.FG_FRACTION
        if self._use_neg:
            # use both pos and neg samples
            n_sel_pos = min(n_pos, n_pos_per_image)
            n_sel_neg = min(batch_size - n_sel_pos, n_neg)
        else:
            # use only pos samples
            n_sel_pos = min(n_pos, batch_size)
            n_sel_neg = 0
        n_sel_pos = int(n_sel_pos)
        n_sel_neg = int(n_sel_neg)

        if DEBUG:
            print 'n_pos:', n_pos
            print 'n_neg:', n_neg
            print 'n_sel_pos:', n_sel_pos
            print 'n_sel_neg:', n_sel_neg

        out_labels = np.empty((rois_labels.shape[0], 1), dtype=np.float32)
        out_labels.fill(-1)
        if n_sel_pos:
            pos_sel_inds = npr.choice(pos_inds, size=n_sel_pos, replace=False)
            if self._use_neg:
                # fg/bg
                out_labels[pos_sel_inds] = rois_labels[pos_sel_inds]
            else:
                # use labels
                out_labels[pos_sel_inds] = \
                    extra_labels[gt_assignment[pos_sel_inds, 0]]
        if n_sel_neg:
            neg_sel_inds = npr.choice(neg_inds, size=n_sel_neg, replace=False)
            out_labels[neg_sel_inds] = rois_labels[neg_sel_inds]

        out_labels = out_labels.astype(np.float32)
        top[0].reshape(*(out_labels.shape))
        top[0].data[...] = out_labels

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        pass
