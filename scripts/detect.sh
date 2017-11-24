#!/bin/sh

tools/detect.py --model models/roialign/deploy.prototxt \
  --weights output/roialign_iter_50000.caffemodel.h5 \
  --gpu -1 --save 164_1154_det.pdf \
  data/ftdata/JPEGImages/164_1154.jpg
