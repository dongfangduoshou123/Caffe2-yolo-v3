# Caffe2-yolo-v3
A Caffe2 implementation of a YOLO v3 Object Detector

This repository contains code for a object detector based on YOLOv3: An Incremental Improvement, implementedin Caffe2. The code is based on the official code of YOLO v3.

Now Only support Inference with CUDA support, train is not Provided yet.

Requirements

    Python 2.7
    OpenCV
    Caffe2

Intall the YOLOModule

  Assume you have already installed the Caffe2 from source.
  
      1. Pelase Use the newest caffe2 in pytorch repo,cut the Yolo folder to the pytorch-master/modules,the open the CMakeList
      in the modules folder and add a line with content add_subdirectory(Yolo),and close it.
      
      (Departured:Make the yolo op as a module:If your Caffe2 older one(https://github.com/caffe2/caffe2),cut the Yolo folder 
      to the Caffe2-master/modules;if your Caffe2 is newer(https://github.com/pytorch/pytorch)cut the Yolo folder to the 
      pytorch-master/modules,the open the CMakeList in the modules folder and add a line with content add_subdirectory(Yolo),
      and close it.)

      2.recompile the whole Caffe2 Project:assume Curent dir is Caffe2-master

            mkdir build

            cd build

            cmake ..

            make install -j8


       3.If success, default in the /usr/local/lib/,we could see generate a so file named libcaffe2_yolo_ops_gpu.so.

The init_net.pb and predict_net.pb is convert from the original model https://pjreddie.com/media/files/yolov3.weights.
The Model file download link is: https://pan.baidu.com/s/1ykYOJgMVXlgACXMC5jAMOQ Password：2z6t

CPPDemo:In the cppDemo folder
please edit four variable init_net\predict_net\coconame_path\test_img as your real path in the main.cpp.

        std::string init_net = "/opt/caffe2_yolov3/init_net.pb";

        std::string predict_net = "/opt/caffe2_yolov3/predict_net.pb";

        std::string coconame_path = "/home/yoloCaffe2/cppDemo/coco.names";

        std::string test_img = "/home/yoloCaffe2/cppDemo/dog.jpg";
  
PythonDemo: demo.py demo1.py

Has problem with using, please make a issue.

