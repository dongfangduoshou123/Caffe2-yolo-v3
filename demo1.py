#-*- coding: UTF-8 -*-
#! /usr/bin/env python
"""
Reads Darknet53 config and weights and creates Caffe2 model

"""
import argparse
import configparser
import io
import os
import sys
import logging
import time
import yaml
from collections import defaultdict

import cv2

from caffe2.python import core
from caffe2.python import workspace
from caffe2.python import model_helper,brew

from caffe2.python import dyndep
from PIL import Image, ImageDraw
import numpy as np

logger = logging.getLogger(__name__)
yolo_ops_lib = myutils.get_yolo_ops_lib()
dyndep.InitOpsLibrary(yolo_ops_lib)

img_dir = '/opt/LabelMe2CocoDataset/decoded_img_48349.jpg'
model_dir = "/opt/caffe2_yolov3/yolov3.minidb"
def _main(args):

    num_classes = 80
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    class_names = load_class_names(namesfile)

    with myutils.NamedCudaScope(0):
        workspace.ResetWorkspace()
        predict_net = prepare_prediction_net(model_dir, "minidb")
        print predict_net.Proto()
        workspace.CreateNet(predict_net,overwrite=True)
        print("The blobs in the workspace after loading the model: {}".format(workspace.Blobs()))
        img = cv2.imread(img_dir)
        b, g, r = cv2.split(img)
        rgb_img = cv2.merge([r, g, b])
        sized = cv2.resize(rgb_img, (416, 416), interpolation=cv2.INTER_CUBIC)
        npar = np.array(sized)
        pp = np.ascontiguousarray(np.transpose(npar,[2,0,1])).reshape(1,3,sized.shape[0],sized.shape[1]).astype(np.float32)/255.0
        for i in range(1):
            list_boxes = do_detect(predict_net,pp)
            boxes = list_boxes[0] + list_boxes[1] + list_boxes[2]
            boxes = nms(boxes, 0.4)
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plot_boxes(image, boxes, 'predictions.jpg', class_names)


if __name__ == '__main__':
    core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    workspace.ResetWorkspace()
    _main(parser.parse_args())





















