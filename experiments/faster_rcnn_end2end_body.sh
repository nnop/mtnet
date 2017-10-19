#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/train.log"
exec &> >(tee "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu 1 \
  --solver models/body/solver.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb ftdata_train \
  --iters 70000 \
  --cfg experiments/faster_rcnn_end2end.yml
