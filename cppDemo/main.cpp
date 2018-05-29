#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <caffe2/core/init.h>
#include <caffe2/core/context.h>
#include <caffe2/core/context_gpu.h>
#include <caffe2/utils/proto_utils.h>
#include <caffe2/core/predictor.h>
#include <boxuitl.h>
extern"C"{
#include <dlfcn.h>
}

#define random_(a,b) (rand()%(b-a+1)+a)
using namespace cv;

using namespace std;
using namespace caffe2;

void* dp = dlopen("/usr/local/lib/libcaffe2_yolo_ops_gpu.so",RTLD_LAZY);
std::string init_net = "/opt/caffe2_yolov3/init_net.pb";
std::string predict_net = "/opt/caffe2_yolov3/predict_net.pb";
std::string coconame_path = "/home/wzq/work/yolov3Tocaffe2/yoloCaffe2/cppDemo/coco.names";
std::string test_img = "/home/wzq/work/yolov3Tocaffe2/yoloCaffe2/cppDemo/dog.jpg";
void showDetectRes(const std::vector<boxInfo>& boxes,cv::Mat & src,std::vector<std::string>& classnames);
int main(int argc, char *argv[])
{
    caffe2::GlobalInit(&argc, &argv);
    std::ifstream in;
    in.open(coconame_path);
    std::string line;
    std::vector<std::string>coconames;
    while(in){
        std::getline(in,line);
        if(line.size())
            coconames.push_back(line);
    }
    std::cout << "coconame size:" << coconames.size() << std::endl;
    cv::Mat image = cv::imread(test_img);
    std::vector<cv::Mat>bgr(3);
    cv::split(image,bgr);
    cv::Mat rgb;

    cv::merge(std::vector<cv::Mat>({bgr[2],bgr[1],bgr[0]})
              ,rgb);
    cv::Size scale(416,416);
    cv::resize(rgb,rgb,scale,0,0,CV_INTER_CUBIC);

    rgb.convertTo(rgb,CV_32FC3,1.0/255);
    std::vector<cv::Mat> channels(3);
    cv::split(rgb,channels);
    std::vector<float> data;
    for(auto& c: channels) {
        data.insert(data.end(),(float*)c.datastart,(float*)c.dataend);
    }
    std::vector<TIndex>dims({1,rgb.channels(),rgb.rows,rgb.cols});
    TensorCPU tensor(dims,data,NULL);

    NetDef initnet,predictnet;
    CAFFE_ENFORCE(ReadProtoFromFile(init_net,&initnet));
    CAFFE_ENFORCE(ReadProtoFromFile(predict_net,&predictnet));
    initnet.mutable_device_option()->set_device_type(CUDA);
    predictnet.mutable_device_option()->set_device_type(CUDA);
    initnet.mutable_device_option()->set_cuda_gpu_id(0);
    predictnet.mutable_device_option()->set_cuda_gpu_id(0);

    Workspace workspace("tmp");
    caffe2::CUDAContext ctx;

    CAFFE_ENFORCE(workspace.RunNetOnce(initnet));
    workspace.CreateNet(predictnet);
    auto input = workspace.CreateBlob("gpu_0/data")->GetMutable<TensorCUDA>();
    input->ResizeLike(tensor);
    ctx.template Copy<float,CPUContext,CUDAContext>(
                tensor.size(),tensor.template data<float>(),input->template mutable_data<float>());
    CAFFE_ENFORCE(workspace.RunNet(predictnet.name()));
    YOLOBoxParser parser;
    std::vector<boxInfo> boxes = parser.parerToHost(workspace);

    showDetectRes(boxes,image,coconames);

    return 0;
}

void showDetectRes(const std::vector<boxInfo> &boxes, Mat &src,std::vector<std::string>& classnames){
    const int width = src.cols;
    const int height = src.rows;
    for(int i = 0;i < boxes.size();i ++){
        boxInfo box = boxes[i];
        float x1 = (box.cx - box.w/2.0)*width;
        float y1 = (box.cy - box.h/2.0)*height;
        float x2 = (box.cx + box.w/2.0)*width;
        float y2 = (box.cy + box.h/2.0)*height;
        std::cout << x1 << " " << y1 << " " << x2 <<" " << y2 << std::endl;
        int r = random_(50,150);
        int g = random_(50,150);
        int b = random_(50,150);
        cv::Scalar rgb(r,g,b);
        float cls_conf = box.cls_max_conf;
        int cls_id = box.cls_max_id;
        int classes = classnames.size();
        std::stringstream ss;
        ss << classnames[cls_id] << "   " << cls_conf;
        std::string text = ss.str();
        ss.str("");
        int font_face = cv::FONT_HERSHEY_COMPLEX;
        double font_scale = 0.5;
        cv::putText(src,text,cv::Point(int(x1),int(y1)),font_face,font_scale,cv::Scalar(0,0,255),1,8);
        cv::rectangle(src,cv::Rect(x1,y1,x2-x1,y2-y1),rgb,1,8,0);
    }
    cv::imshow("res",src);
    cv::waitKey(0);

}




























