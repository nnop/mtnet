# Data directory structure:
#   ftdata/
#     |- JPEGImages/
#     |- Annotations/
#     |- ImageSets/
#     `- results/
        
import sys
import os
import json
import logging
from datasets.imdb import imdb
import numpy as np
import cPickle
import PIL

from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import clip_boxes

class ftdata(imdb):
    def __init__(self, image_set, data_path=None):
        imdb.__init__(self, 'ftdata_' + image_set)
        self._image_set = image_set
        self._data_path = self._get_default_path() if data_path is None \
                            else data_path
        self._body_classes = ('listen', 'write', 'handup',
                              'positive', 'negative')
        self._head_classes = ('smile', 'openmouth', 'netural',
                              'bowhead', 'twist', 'other')
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb
        self.config = {}
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def append_flipped_images(self):
        num_images = self.num_images
        widths = self._get_widths()
        for i in xrange(num_images):
            entry = self.roidb[i].copy()    # shallow copy
            body_boxes = self._flip_boxes(widths[i], self.roidb[i]['body_boxes'])
            assert (body_boxes[:, 2] >= body_boxes[:, 0]).all()
            head_boxes = self._flip_boxes(widths[i], self.roidb[i]['head_boxes'])
            assert (head_boxes[:, 2] >= head_boxes[:, 0]).all()
            entry['body_boxes'] = body_boxes
            entry['head_boxes'] = head_boxes
            entry['flipped'] = True
            self.roidb.append(entry)
        self._image_index = self._image_index * 2

    def _flip_boxes(self, im_wid, boxes):
        flipped_boxes = boxes.copy()
        flipped_boxes[:, 0] = im_wid - boxes[:, 2] - 1
        flipped_boxes[:, 2] = im_wid - boxes[:, 0] - 1
        return flipped_boxes

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where ftdata is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'ftdata')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_annotation(index) for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _load_annotation(self, index):
        """
        Load image and bounding boxes info from json file.
        Info:
            - image
            - width
            - height
            - stu_classes
            - body_boxes
            - body_classes
            - head_boxes
            - head_classes
        """
        # image info
        im_path = self.image_path_from_index(index)
        im_wid, im_hei = PIL.Image.open(im_path).size
        # anno info
        filename = os.path.join(self._data_path, 'Annotations', index + '.json')
        with open(filename) as f:
            anno_info = json.load(f)
        objs = anno_info['persons']
        num_objs = len(objs)

        # student
        stu_classes = np.empty((num_objs), dtype=np.int32)
        stu_classes.fill(-1)
        # body
        body_boxes = np.zeros((num_objs, 4), dtype=np.float32)
        body_classes = np.empty((num_objs), dtype=np.int32)
        body_classes.fill(-1)
        # head
        head_boxes = np.zeros((num_objs, 4), dtype=np.float32)
        head_classes = np.empty((num_objs), dtype=np.int32)
        head_classes.fill(-1)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # student
            stu_classes[ix] = int(obj['student'])
            # body
            body_label = obj['body']['label']
            if body_label in self._body_classes:
                body_classes[ix] = self._body_classes.index(body_label)
            body_boxes[ix, :] = obj['body']['bbox']
            # head
            try:
                head_label = obj['head']['label']
            except TypeError as e:
                # head is null
                continue
            assert head_label in self._head_classes, \
                    'Wrong head label: {}'.format(head_label)
            head_classes[ix] = self._head_classes.index(head_label)
            head_boxes[ix, :] = obj['head']['bbox']
            # clipboxes
            body_boxes = clip_boxes(body_boxes, (im_hei, im_wid))
            head_boxes = clip_boxes(head_boxes, (im_hei, im_wid))

        return {'image': im_path, 'width': im_wid, 'height': im_hei,
                'stu_classes': stu_classes,
                'body_boxes': body_boxes,
                'body_classes': body_classes,
                'head_boxes': head_boxes,
                'head_classes': head_classes,
                'flipped' : False}

