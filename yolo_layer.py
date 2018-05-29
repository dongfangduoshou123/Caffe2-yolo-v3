import argparse
import configparser
import io
import os
import logging
import six
import sys
import time
from collections import defaultdict

from caffe2.python import core
from caffe2.python import workspace
from caffe2.python.predictor.predictor_exporter import *
from numpy import *
import numpy as np

import myutils

def get_region_boxes(tx,ty,tw,th,det_confs,class_prob,conf_thresh,num_classes,num_anchors,only_objectness=1):
    batch = tx.shape[0]
    h = tx.shape[2]
    w = tx.shape[3]


    tx = tx.reshape(-1)

    cls_confs = class_prob.reshape(batch*num_anchors,num_classes,h*w)
    # print cls_confs
    cls_max_confs = cls_confs.max(1)
    cls_max_ids = cls_confs.argmax(axis=1)
    boxes = []
    for b in range(batch):
        for i in range(num_anchors):
            for cx in range(w):
                for cy in range(h):
                    det_conf = det_confs[i*w*h + cx*h + cy]
                    if only_objectness:
                        conf = det_confs[i*w*h + cx*h + cy]
                    else:
                        conf = det_confs[i*w*h + cx*h + cy] * cls_max_confs[i][cx*h+ cy]

                    if conf > conf_thresh:
                        bcx = tx[i*w*h + cx*h + cy]
                        bcy = ty[i*w*h + cx*h + cy]
                        bw = tw[i*w*h + cx*h + cy]
                        bh = th[i*w*h + cx*h + cy]
                        cls_max_conf = cls_max_confs[i][cx*h+ cy]
                        cls_max_id = cls_max_ids[i][cx*h+ cy]
                        box = [bcx / w, bcy / h, bw / w, bh / h, det_conf, cls_max_conf, cls_max_id]
                        # print bcx / w, bcy / h, bw / w ," cx"
                        boxes.append(box)
    return boxes


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def do_detect(model,im):
    anchor_mask1 = [6,7,8]
    anchor_mask2 = [3,4,5]
    anchor_mask3 = [0,1,2]
    anchors_mask = [anchor_mask1,anchor_mask2,anchor_mask3]
    anchors1 = ['10','13',  '16','30',  '33','23',  '30','61',  '62','45',  '59','119',  '116','90',  '156','198',  '373','326']
    anchors = [float(i) for i in anchors1]
    numclass = 80
    num_anchors = 9
    anchor_step = len(anchors)/num_anchors
    stride = [32,16,8]
    start = time.time()
    # with myutils.NamedCudaScope(0):
    workspace.FeedBlob(core.ScopedName('data'), im)
    workspace.RunNet(model.Proto().name)
    tx = []
    tx.append(workspace.FetchBlob(core.ScopedName("tx_1")))
    tx.append(workspace.FetchBlob(core.ScopedName("tx_2")))
    tx.append(workspace.FetchBlob(core.ScopedName("tx_3")))
    ty = []
    ty.append(workspace.FetchBlob(core.ScopedName("ty_1")))
    ty.append(workspace.FetchBlob(core.ScopedName("ty_2")))
    ty.append(workspace.FetchBlob(core.ScopedName("ty_3")))
    tw = []
    tw.append(workspace.FetchBlob(core.ScopedName("tw_1")))
    tw.append(workspace.FetchBlob(core.ScopedName("tw_2")))
    tw.append(workspace.FetchBlob(core.ScopedName("tw_3")))
    th = []
    th.append(workspace.FetchBlob(core.ScopedName("th_1")))
    th.append(workspace.FetchBlob(core.ScopedName("th_2")))
    th.append(workspace.FetchBlob(core.ScopedName("th_3")))
    det_conf = []
    det_conf.append(workspace.FetchBlob(core.ScopedName("det_conf_1")))
    det_conf.append(workspace.FetchBlob(core.ScopedName("det_conf_2")))
    det_conf.append(workspace.FetchBlob(core.ScopedName("det_conf_3")))
    class_prob = []
    class_prob.append(workspace.FetchBlob(core.ScopedName("class_prob_1")))
    class_prob.append(workspace.FetchBlob(core.ScopedName("class_prob_2")))
    class_prob.append(workspace.FetchBlob(core.ScopedName("class_prob_3")))
    finish = time.time()
    all_boxes = []
    for i in range(len(tx)):
        masked_anchors = []
        for m in anchors_mask[i]:
            masked_anchors += anchors[m*anchor_step:(m+1)*anchor_step]
        masked_anchors = [anchor/stride[i] for anchor in masked_anchors]
        boxes = \
            get_region_boxes(
                tx[i],
                ty[i],
                tw[i],
                th[i],
                det_conf[i],
                class_prob[i],
                0.6,
                numclass
                , len(anchors_mask[i]),
            )
        all_boxes.append(boxes)

    return all_boxes

