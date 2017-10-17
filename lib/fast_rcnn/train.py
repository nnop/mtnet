# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from utils.timer import Timer
import numpy as np
import os
import logging
from caffe.proto import caffe_pb2
from google.protobuf import text_format

from fast_rcnn.config import cfg
from datasets.ftdata import ftdata

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

        assert (cfg.TRAIN.HAS_RPN \
            and cfg.TRAIN.BBOX_REG \
            and cfg.TRAIN.BBOX_NORMALIZE_TARGETS \
            and cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED)

        self.solver = caffe.SGDSolver(solver_prototxt)
        # copy pretrained weights
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb)

    def _unnormalize_bbox_params(self, blob_name, num_classes):
        net = self.solver.net
        orig_0 = net.params[blob_name][0].data.copy()
        orig_1 = net.params[blob_name][1].data.copy()

        bbox_means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS),
                (num_classes, 1))
        bbox_stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS),
                (num_classes, 1))

        # scale and shift with bbox reg unnormalization; then save snapshot
        net.params[blob_name][0].data[...] = \
                (net.params[blob_name][0].data * bbox_stds[:, np.newaxis])
        net.params[blob_name][1].data[...] = \
                (net.params[blob_name][1].data * bbox_stds + self.bbox_means)

        return orig_0, orig_1

    def _restore_bbox_params(self, blob_name, orig_weights):
        assert len(orig_weights) == 2
        net = self.solver.net
        net.params[blob_name][0].data[...] = orig_weights[0]
        net.params[blob_name][1].data[...] = orig_weights[1]

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net
        blob_name = 'bbox_pred/body'
        num_body_poses = len(ftdata._pose_classes)
        orig_weights = self._unnormalize_bbox_params(blob_name, num_body_poses)

        # save original values
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        # restore net to original state
        self._restore_bbox_params(blob_name, orig_weights)
        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())

        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'
    return imdb.roidb

def train_net(solver_prototxt, roidb, output_dir,
              pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""

    sw = SolverWrapper(solver_prototxt, roidb, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    model_paths = sw.train_model(max_iters)
    print 'done solving'
    return model_paths
