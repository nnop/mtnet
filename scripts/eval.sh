#!/bin/sh

tools/test_net.py --imdb ftdata_test --gpu -1 \
  --model models/roialign/deploy.prototxt \
  --weights output/roialign_iter_50000.caffemodel.h5 \
  --vis-dir output/vis_out/ \
  --gt-dir data/ftdata/Annotations/ \
  --det-dir output/det_dump/
