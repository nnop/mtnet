#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import cv2
import os.path as osp
import sys
import argparse

def draw_bbox(ax, bbox, text='', color='r', linewidth=1):
    xmin, ymin, xmax, ymax = bbox
    coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
    ax.add_patch(plt.Rectangle(*coords, fill=False, 
            edgecolor=color, linewidth=linewidth))
    ax.text(xmin, ymin, text, bbox={'facecolor': color, 'alpha': 0.5})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--json', required=True)
    args = parser.parse_args()

    image_path = args.image
    json_path = args.json

    im = cv2.imread(image_path)[:, :, [2, 1, 0]]
    plt.figure(figsize=(15, 10))
    plt.imshow(im)
    with open(json_path) as f:
        info = json.load(f)
    ax = plt.gca()
    for i, psn in enumerate(info['persons']):
        body_bbox = psn['body']['bbox']
        body_pose = psn['body']['label']
        draw_bbox(ax, body_bbox, text=body_pose[:3], color='r', linewidth=2)
        try:
            head_bbox = psn['head']['bbox']
            head_pose = psn['head']['label']
            draw_bbox(ax, head_bbox, text=head_pose[:3], color='g', linewidth=1)
        except TypeError as e:
            print 'body[{}] has no head.'.format(i)
    save_fn = osp.splitext(osp.split(image_path)[1])[0] + '_vis.png'
    plt.savefig(save_fn)
    print 'save to', save_fn
