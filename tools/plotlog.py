#!/usr/bin/env python
from pprint import pprint
import sys
import re
from os import path
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython import embed

FLOAT_PTN = r'\d+\.[\de-]+'
INT_PTN = r'\d+(?:,\d+)*'

# SOLVING_START = re.compile(r'solver\.cpp:\d+] Solving')
SOLVING_START = re.compile(r'Iteration 0')

# test output
BLOCK_START_TE = re.compile(r'Iteration %s, Testing net' % (INT_PTN,))
RE_DICT_TE = {
    'iter': re.compile(r'Iteration (%s), Testing net' % INT_PTN),
    'mAP': re.compile(r'Test net output #\d: detection_eval = (%s)' % FLOAT_PTN)
}
PLOT_KEYS_TE = ['mAP']

# train output
BLOCK_START_TR = re.compile(r'Iteration %s .+, loss = %s' % (INT_PTN, FLOAT_PTN))
RE_DICT_TR = {
    'iter': re.compile(r'Iteration (%s) .+, loss = %s' % (INT_PTN, FLOAT_PTN)),
    'loss': re.compile(r'Iteration %s .+, loss = (%s)' % (INT_PTN, FLOAT_PTN)),
    # 'rpn_cls_loss': re.compile(r'Train net output #\d: rpn_cls_loss = (%s)' % FLOAT_PTN),
    # 'rpn_loss_bbox': re.compile(r'Train net output #\d: rpn_loss_bbox = (%s)' % FLOAT_PTN),
}

PLOT_KEYS_TR = ['loss']

def getTrainingLog(logfile):
    with open(logfile) as f:
        for line in f:
            if SOLVING_START.search(line):
                break
        all_lines = list(f)
    return all_lines

def parseBlock(blk_lines, RE_DICT):
    """
    parse one single block into a dict record
    return `None` if there are some missing results
    """
    record = {}
    for re_key in RE_DICT:
        record[re_key] = None

    for l in blk_lines:
        for re_key in RE_DICT:
            try:
                re_val = RE_DICT[re_key].search(l).group(1)
                record[re_key] = parseValue(re_val)
            except AttributeError as e:
                continue
    for k in RE_DICT:
        if record[k] is None:
            return None
    return record

def getBlocks(all_lines, BLOCK_START):
    blk_idx = []
    for idx, l in enumerate(all_lines):
        if BLOCK_START.search(l):
            blk_idx.append(idx)
    blk_idx.append(len(all_lines))
    blk_idx = np.array(blk_idx)

    blk_beg_idx = blk_idx[:-1]
    blk_end_idx = blk_idx[1:]

    num_blocks = len(blk_idx)-1

    blocks = [all_lines[blk_beg_idx[i]:blk_end_idx[i]] for i in range(num_blocks)]
    return blocks

def parseValue(val_str):
    val_str = ''.join(val_str.split(','))
    try:
        return int(val_str)
    except ValueError as e:
        return float(val_str)

def dict2array(records):
    """
    parse the records into an array
    """
    dict_arr = {}
    for rec in records:
        # add values
        for k in rec:
            if k in dict_arr:
                dict_arr[k].append(rec[k])
            else:
                dict_arr[k] = [rec[k]]
    return dict_arr


if __name__ == '__main__':
    if (len(sys.argv) != 2):
        print 'Usage: %s log_file' % (path.split(sys.argv[0])[1], )
        sys.exit(1)

    logfile = sys.argv[1]
    all_lines = getTrainingLog(logfile)

    # get tr blocks
    blocks_tr = getBlocks(all_lines, BLOCK_START_TR)
    records_tr = [parseBlock(blk, RE_DICT_TR) for blk in blocks_tr]
    records_tr = [r for r in records_tr if r]
    res_tr = dict2array(records_tr)

    plt.figure(figsize=(12, 8))
    ax1 = plt.gca()
    ax1.plot(res_tr['iter'], res_tr['loss'], label='iter', color='r')
    ax1.legend(shadow=True, loc=(0.6, 0.4))
    plt.xlabel('iter')
    plt.ylabel('loss')

    # # get te blocks
    # blocks_te = getBlocks(all_lines, BLOCK_START_TE)
    # records_te = [parseBlock(blk, RE_DICT_TE) for blk in blocks_te]
    # records_te = [r for r in records_te if r]
    # res_te = dict2array(records_te)

    # fig = plt.figure()
    # ax1 = plt.gca()
    # for lab in PLOT_KEYS_TE:
    #     try:
    #         ax1.plot(res_te['iter'], res_te[lab], label='te:'+lab,
    #                 color=npr.random(3))
    #     except KeyError as e:
    #         continue
    # ax1.legend(shadow=True, loc=(0.6, 0.4))
    #
    # ax2 = ax1.twinx()
    # for lab in PLOT_KEYS_TR:
    #     ax2.plot(res_tr['iter'], res_tr[lab], label='tr:'+lab,
    #             color=npr.random(3))
    # plt.gca().legend(shadow=True, loc=(0.6, 0.6))
    # # plt.ylim([0., 1.])
    # plt.xlabel('iter')

    plt.ylim([min(res_tr['loss']), 10.])
    plt.savefig('log.png')

    print 'done.'
