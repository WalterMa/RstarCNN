# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) Microsoft. All rights reserved.
# Written by Ross Girshick, 2015.
# Licensed under the BSD 2-clause "Simplified" license.
# See LICENSE in the project root for license information.
# --------------------------------------------------------
# --------------------------------------------------------
# R*CNN
# Written by Georgia Gkioxari, 2015.
# See LICENSE in the project root for license information.
# --------------------------------------------------------
# --------------------------------------------------------
# Written by Wentao Ma, 2018.
# --------------------------------------------------------

import datasets.pascal_voc
import os
import datasets.imdb
import json
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess

class loar(datasets.imdb):
    def __init__(self, image_set, devkit_path=None):
        datasets.imdb.__init__(self, 'loar_' + image_set)
        self._image_set = image_set
        self._root_path = self._get_default_path() if devkit_path is None \
                            else os.path.expanduser(devkit_path)
        self._classes = ('bendover', 'jump', 'lying', 'run',
                         'sit', 'squat', 'stand', 'stretch', 'walk')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True}

        assert os.path.exists(self._root_path), \
                'Root path does not exist: {}'.format(self._root_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._root_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._root_path, 'ImageSets', self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'loar')

    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(self._root_path, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

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

        # Load all annotation file data (should take < 30 s).
        gt_roidb = [self._load_loar_annotation(index)
                    for index in self.image_index]

        # print number of ground truth classes
        cc = np.zeros(len(self._classes), dtype=np.int16)
        for i in xrange(len(gt_roidb)):
            for n in xrange(len(gt_roidb[i]['gt_classes'])):
                cc[gt_roidb[i]['gt_classes'][n]] += 1

        for ic, nc in enumerate(cc):
            print "Count {:s} : {:d}".format(self._classes[ic], nc)

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                 self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_roidb(gt_roidb)
        roidb = self._merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _merge_roidbs(self, a, b):
        assert len(a) == len(b)
        for i in xrange(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                            b[i]['gt_classes']))
            a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                       b[i]['gt_overlaps']])
        return a

    def _load_selective_search_roidb(self, gt_roidb):

        filename = os.path.join(self._root_path, 'selective_search',
                                self.name + '.mat')

        # filename = op.path.join(self.cache_path, 'MCG_data', self.name + '.mat')

        assert os.path.exists(filename), \
                'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)

        num_images = raw_data['boxes'].ravel().shape[0]
        ss_roidb = []
        for i in xrange(num_images):
            boxes = raw_data['boxes'].ravel()[i][:, (1, 0, 3, 2)] - 1
            num_boxes = boxes.shape[0]
            gt_boxes = gt_roidb[i]['boxes']
            gt_classes = gt_roidb[i]['gt_classes']
            gt_overlaps = \
                    utils.cython_bbox.bbox_overlaps(boxes.astype(np.float),
                                                    gt_boxes.astype(np.float))
            argmaxes = gt_overlaps.argmax(axis=1)
            maxes = gt_overlaps.max(axis=1)
            I = np.where(maxes > 0)[0]
            overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)
            overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            overlaps = scipy.sparse.csr_matrix(overlaps)
            ss_roidb.append({'boxes' : boxes,
                             'gt_classes' : -np.ones((num_boxes,),
                                                      dtype=np.int32),
                             'gt_overlaps' : overlaps,
                             'flipped' : False})
        return ss_roidb

    def _load_loar_annotation(self, index):
        """
        Load image and bounding boxes info from JSON file
        """
        filename = os.path.join(self._root_path, 'Annotations', index + '.json')
        # print 'Loading: {}'.format(filename)

        with open(filename) as f:
            anno = json.load(f)

        num_objs = len(anno['persons'])
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        for i, p in enumerate(anno['persons']):
            cls = p['actions']
            cls_names = [key for key, value in cls.items() if value == 1]
            assert len(cls_names) == 1, 'Only support mutually exclusive classes yet.'
            cls_id = self._class_to_ind[cls_names[0]]
            xmin = p['bndbox']['xmin']
            xmax = p['bndbox']['xmax']
            ymin = p['bndbox']['ymin']
            ymax = p['bndbox']['ymax']

            boxes[i, :] = [xmin, ymin, xmax, ymax]
            gt_classes[i] = cls_id
            overlaps[i, cls_id] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _ap(self, all_scores, all_labels):
        ap = np.zeros((self.num_classes), dtype=np.float32)

        for a in xrange(self.num_classes):
            tp = all_labels[self.classes[a]] == a
            npos = np.sum(tp, axis=0)
            fp = all_labels[self.classes[a]] != a
            sc = all_scores[self.classes[a]]
            cat_all = np.hstack((tp, fp, sc))
            ind = np.argsort(cat_all[:, 2])
            cat_all = cat_all[ind[::-1], :]
            tp = np.cumsum(cat_all[:, 0], axis=0)
            fp = np.cumsum(cat_all[:, 1], axis=0)

            # # Compute precision/recall
            rec = tp / npos
            prec = np.divide(tp, (fp + tp))
            ap[a] = self.VOCap(rec, prec)

        for cls_ind, cls in enumerate(self.classes):
            print '%s\t= %f' % (cls, ap[cls_ind])
        print 'Mean AP\t= %f' % (np.mean(ap))
        return ap

    def VOCap(self, rec, prec):
        rec = rec.reshape(rec.size, 1)
        prec = prec.reshape(prec.size, 1)
        z = np.zeros((1, 1))
        o = np.ones((1, 1))
        mrec = np.vstack((z, rec, o))
        mpre = np.vstack((z, prec, z))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        # i = find(mrec(2:end)~=mrec(1:end-1))+1;
        I = np.where(mrec[1:] != mrec[0:-1])[0] + 1
        ap = 0
        for i in I:
            ap = ap + (mrec[i] - mrec[i - 1]) * mpre[i]
        return ap

    def _write_voc_results_file(self, all_boxes):

        path = os.path.join(self._root_path, 'results')
        for cls_ind, cls in enumerate(self.classes):
            
            print 'Writing {} VOC results file'.format(cls)
            filename = os.path.join(path, self.name + '_' + cls + '.txt')
            with open(filename, 'wt') as f:
                for i in xrange(all_boxes.shape[0]):
                    ind = all_boxes[i,0].astype(np.int64)
                    index = self.image_index[ind-1]
                    voc_id = all_boxes[i,1].astype(np.int64)
                    score = all_boxes[i,2+cls_ind]
                    
                    f.write('{:s} {:d} {:.3f}\n'.
                            format(index, voc_id, score))           


if __name__ == '__main__':
    d = datasets.loar('trainval', root_path='~/data/LoAR')
    res = d.roidb
    from IPython import embed
    embed()
