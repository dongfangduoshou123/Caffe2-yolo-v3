TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp


INCLUDEPATH +=/home/wzq/work/package/opencv/include\
/home/wzq/work/package/opencv/include/opencv\
/home/wzq/work/package/opencv/include/opencv2



unix:!macx: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_core

unix:!macx: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_highgui

unix:!macx: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_imgproc


unix:!macx: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_imgcodecs


unix:!macx: LIBS += -L$$PWD/../../../../../../usr/lib/x86_64-linux-gnu/ -lpthread

unix:!macx: LIBS += -L$$PWD/../../../../../../usr/local/cuda/lib64/ -lcublas

unix:!macx: LIBS += -L$$PWD/../../../../../../usr/local/cuda/lib64/ -lcudart

unix:!macx: LIBS += -L$$PWD/../../../../../../usr/local/cuda/lib64/ -lcudnn

unix:!macx: LIBS += -L$$PWD/../../../../../../usr/local/cuda/lib64/ -lcufft

unix:!macx: LIBS += -L$$PWD/../../../../../../usr/local/cuda/lib64/ -lcurand
unix:!macx: LIBS += -L$$PWD/../../../../../../usr/local/cuda/lib64/ -lcuda

unix:!macx: LIBS += -L$$PWD/../../../../../../usr/lib/x86_64-linux-gnu/ -lprotobuf

unix:!macx: LIBS += -L$$PWD/../../../../../../usr/lib/x86_64-linux-gnu/ -lglog

INCLUDEPATH += $$PWD/../../../../../../usr/lib/x86_64-linux-gnu \
/usr/local/cuda/include

unix:!macx: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lcaffe2_gpu

unix:!macx: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lcaffe2
unix:!macx: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lcaffe2_yolo_ops_gpu

unix:!macx: LIBS += -L$$PWD/../../../../../../usr/lib/x86_64-linux-gnu/ -ldl

HEADERS += \
    boxuitl.h
