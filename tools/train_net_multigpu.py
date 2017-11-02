#!/usr/bin/env python

"""Train a Faster R-CNN with Multiple GPUs"""

import caffe
import numpy as np
import sys
import logging
from multiprocessing import Process

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from datasets.factory import get_imdb
from utils.logger import config_logger

config_logger()

def solve(gpus, uid, rank, solver_proto, roidb, weights=None, snapshot=None):
    cfg.GPU_ID = gpus[rank]
    # setting for current process
    caffe.set_mode_gpu()
    caffe.set_device(gpus[rank])
    caffe.set_solver_count(len(gpus))
    caffe.set_solver_rank(rank)
    caffe.set_multiprocess(True)

    solver = caffe.SGDSolver(solver_proto)
    logging.info('uid: {}, rank: {}, layer_wise_reduce: {}'
            .format(uid, rank, solver.param.layer_wise_reduce))
    max_iter = solver.param.max_iter
    snapshot_iters = solver.param.snapshot
    if snapshot:
        solver.restore(snapshot)
    if weights:
        solver.net.copy_from(weights)
    solver.net.layers[0].set_roidb(roidb, rank)

    nccl = caffe.NCCL(solver, uid)
    solver.add_callback(nccl)
    if solver.param.layer_wise_reduce:
        solver.net.after_backward(nccl)
    
    nccl.bcast()
    curr_iter = solver.iter
    while curr_iter < max_iter:
        step_iters = snapshot_iters - curr_iter % snapshot_iters
        solver.step(step_iters)
        if rank == 0:
            logging.info('curr_iter: {}, step_iters: {}'.format(curr_iter, step_iters))
            solver.snapshot()
            curr_iter += step_iters

if __name__ == "__main__":
    solver_proto = 'models/multigpu/solver.prototxt'
    weights_file = 'data/imagenet_models/VGG16.v2.caffemodel'
    cfg_file = 'experiments/faster_rcnn_end2end.yml'
    gpus = [3, 4, 5, 6]

    # caffe
    caffe.init_log(0, True)
    caffe.log('Using device {}'.format(str(gpus)))
    uid = caffe.NCCL.new_uid()

    # cfg
    cfg_from_file(cfg_file)
    assert (cfg.TRAIN.HAS_RPN                \
        and cfg.TRAIN.BBOX_REG               \
        and cfg.TRAIN.BBOX_NORMALIZE_TARGETS \
        and cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED)

    # roidb
    imdb_name = 'ftdata_train'
    imdb = get_imdb(imdb_name)
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
    roidb = imdb.roidb

    procs = []
    for rank in range(len(gpus)):
        p = Process(target=solve,
                    args=(gpus, uid, rank, solver_proto, roidb, weights_file))
        p.daemon = False
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
