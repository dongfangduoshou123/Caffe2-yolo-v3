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
#########################
from myutils import *
from yolo_layer import *
logger = logging.getLogger(__name__)
yolo_ops_lib = myutils.get_yolo_ops_lib()
dyndep.InitOpsLibrary(yolo_ops_lib)
model_dir = "/opt/caffe2_yolov3/yolov3.minidb"
def main():
    num_classes = 80
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    class_names = load_class_names(namesfile)
    cap = cv2.VideoCapture()
    cap.open(0)
    ret,img = cap.read()
    if ret == False:
        print "Camera open failed!"
        return
    with myutils.NamedCudaScope(0):
        workspace.ResetWorkspace)
        predict_net = prepare_prediction_net(model_dir, "minidb")
        workspace.CreateNet(predict_net,overwrite=True)
        while(True):
            ret, img = cap.read()
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
            res = myutils.plot_boxes(image, boxes,"predict.jpg", class_names)


if __name__ == '__main__':
    core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    workspace.ResetWorkspace()
    main()
