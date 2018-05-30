# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import math
from PIL import Image, ImageDraw
import numpy as np
import cv2

from six import string_types
import contextlib

from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import dyndep
from caffe2.python import scope



# Default value of the CMake install prefix
_CMAKE_INSTALL_PREFIX = '/usr/local'


def get_runtime_dir():
    """Retrieve the path to the runtime directory."""
    return sys.path[0]


def get_py_bin_ext():
    """Retrieve python binary extension."""
    return '.py'


def set_up_matplotlib():
    """Set matplotlib up."""
    import matplotlib
    # Use a non-interactive backend
    matplotlib.use('Agg')


def exit_on_error():
    """Exit from a detectron tool when there's an error."""
    sys.exit(1)


def import_nccl_ops():
    """Import NCCL ops."""
    # There is no need to load NCCL ops since the
    # NCCL dependency is built into the Caffe2 gpu lib
    pass


def get_yolo_ops_lib():
    """Retrieve Detectron ops library."""
    # Candidate prefixes for the detectron ops lib path
    prefixes = [_CMAKE_INSTALL_PREFIX, sys.prefix, sys.exec_prefix] + sys.path
    # Search for detectron ops lib
    for prefix in prefixes:
        ops_path = os.path.join(prefix, 'lib/libcaffe2_yolo_ops_gpu.so')
        if os.path.exists(ops_path):
            # TODO(ilijar): Switch to using a logger
            print('Found Yolo ops lib: {}'.format(ops_path))
            break
    assert os.path.exists(ops_path), \
        ('Yolo ops lib not found; make sure that your Caffe2 '
         'version includes Yolo module')
    return ops_path

@contextlib.contextmanager
def NamedCudaScope(gpu_id):
    """Creates a GPU name scope and CUDA device scope. This function is provided
    to reduce `with ...` nesting levels."""
    with GpuNameScope(gpu_id):
        with CudaScope(gpu_id):
            yield


@contextlib.contextmanager
def GpuNameScope(gpu_id):
    """Create a name scope for GPU device `gpu_id`."""
    with core.NameScope('gpu_{:d}'.format(gpu_id)):
        yield


@contextlib.contextmanager
def CudaScope(gpu_id):
    """Create a CUDA device scope for GPU device `gpu_id`."""
    gpu_dev = CudaDevice(gpu_id)
    with core.DeviceScope(gpu_dev):
        yield


@contextlib.contextmanager
def CpuScope():
    """Create a CPU device scope."""
    cpu_dev = core.DeviceOption(caffe2_pb2.CPU)
    with core.DeviceScope(cpu_dev):
        yield

def CudaDevice(gpu_id):
    """Create a Cuda device."""
    return core.DeviceOption(caffe2_pb2.CUDA, gpu_id)

def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea

def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes
    # det_confs = torch.zeros(len(boxes))
    det_confs = np.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1-boxes[i][4]

    sortIds = np.argsort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    box_j[4] = 0
    return out_boxes
def show(img,boxes,class_names=None):
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]])
    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)
    width = img.shape[0]
    height = img.shape[1]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2] / 2.0) * width
        y1 = (box[1] - box[3] / 2.0) * height
        x2 = (box[0] + box[2] / 2.0) * width
        y2 = (box[1] + box[3] / 2.0) * height

        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)
            # draw.text((x1, y1), class_names[cls_id], fill=rgb)
        # draw.rectangle([x1, y1, x2, y2], outline=rgb)
        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),rgb,2,8,1)

def plot_boxes(img, boxes, savename=None, class_names=None):
    colors = np.array([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])
    # img = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)

    rgb = (100, 120, 234)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2]/2.0) * width
        y1 = (box[1] - box[3]/2.0) * height
        x2 = (box[0] + box[2]/2.0) * width
        y2 = (box[1] + box[3]/2.0) * height
        print (x1," " ,y1 ," ",x2 ," ",y2)
        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            # print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw.text((x1, y1), class_names[cls_id], fill=rgb)
        draw.rectangle([x1, y1, x2, y2], outline = rgb)
        draw.rectangle([0, 0, 352, 288], outline=rgb)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    sh = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    cv2.imshow("demo", sh)
    cv2.waitKey(5)
    return img

def UnscopeName(possibly_scoped_name):
    """Remove any name scoping from a (possibly) scoped name. For example,
    convert the name 'gpu_0/foo' to 'foo'."""
    assert isinstance(possibly_scoped_name, string_types)
    return possibly_scoped_name[
        possibly_scoped_name.rfind(scope._NAMESCOPE_SEPARATOR) + 1:]
