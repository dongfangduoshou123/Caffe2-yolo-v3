#ifndef CAFFE2_OPERATORS_YOLO_OP_H_
#define CAFFE2_OPERATORS_YOLO_OP_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
namespace caffe2 {

struct MaxConfIndex{
    float conf;
    int index;
};

struct boxInfo{
    float cx;
    float cy;
    float h;
    float w;
    float det_conf;
    float cls_max_conf;
    float cls_max_id;
};



template <typename T,class Context>
class YoloOp final:public Operator<Context> {
public:
    USE_OPERATOR_CONTEXT_FUNCTIONS;
    YoloOp(const OperatorDef& operator_def, Workspace* ws)
        : Operator<Context>(operator_def, ws),
          anchor_mask_(OperatorBase::GetRepeatedArgument<int>("anchor_mask")),
          anchors_(OperatorBase::GetRepeatedArgument<int>("anchors")),
          numclass_(OperatorBase::GetSingleArgument<int>("numclass",80)),
          numanchors_(OperatorBase::GetSingleArgument<int>("numanchors",9)),
          stride_(OperatorBase::GetSingleArgument<int>("stride",16)),
          is_train(OperatorBase::GetSingleArgument<bool>("is_train",false)),
          conf_thresh_(OperatorBase::GetSingleArgument<float>("conf_thresh",0.6)){
          anchor_step_ = anchors_.size() / numanchors_;
          for(int i = 0;i < anchor_mask_.size();i++){
              float x = float(anchors_[anchor_mask_[i]*anchor_step_])/this->stride_;
              float y = float(anchors_[anchor_mask_[i]*anchor_step_+1])/this->stride_;
              masked_anchor_.push_back(x);
              masked_anchor_.push_back(y);
          }
    }

    bool RunOnDevice() override;
    void SetDeviceTensor(const std::vector<float>& data, Tensor* tensor) {
        tensor->Resize(data.size());
        context_.template Copy<float, CPUContext, Context>(
        data.size(), data.data(), tensor->template mutable_data<float>());
    }
    void GetTensorToHost(const Tensor* tensor,std::vector<float>& data ) {
        data.resize(tensor->size());
        context_.template Copy<float,Context,CPUContext>(
                    data.size(),tensor->template data<float>(),data.data());

    }
    void computeMaxConfIndex(Tensor* tensor){
        std::vector<float>data;
        GetTensorToHost(tensor,data);
        maxConfIndex_.resize(tensor->dim(0));
        for(int i = 0;i < maxConfIndex_.size();i++){
            std::vector<MaxConfIndex> cur;
            cur.resize(tensor->dim(2));
            int offset = tensor->dim(2)*this->numclass_;
            for(int j = 0;j < this->numclass_;j ++){

                for(int k = 0;k < tensor->dim(2);k ++){
                    if(j == 0){
                        cur[k].conf = data[i*offset + j*tensor->dim(2) + k];
                        cur[k].index = 0;
                    }else{
                        float conf = data[i*offset + j*tensor->dim(2) + k];
                        if (conf > cur[k].conf){
                            cur[k].conf = conf;
                            cur[k].index = j;
                        }
                    }
                }
            }
            maxConfIndex_[i] = cur;
        }
    }

    void GetYoloBoxes(){
        this->boxes_.resize(0);
        for(int b = 0;b < this->batch_;b ++){
            for(int i = 0;i < this->anchor_mask_.size();i ++){
                for(int cx = 0;cx < this->w_;cx++){
                    for(int cy = 0;cy < this->h_;cy ++){
                        float det_conf = this->det_conf_[i*this->w_*this->h_ + cx*this->h_ + cy];
                        float conf = det_conf;
                        this->conf_thresh_ = 0.6;
                        if(conf > conf_thresh_){
                            boxInfo box;
                            box.cx = this->tx_[i*this->w_*this->h_ + cx*this->h_ + cy]/this->w_;
                            box.cy = this->ty_[i*this->w_*this->h_ + cx*this->h_ + cy]/this->h_;
                            box.w = this->tw_[i*this->w_*this->h_ + cx*this->h_ + cy]/this->w_;
                            box.h = this->th_[i*this->w_*this->h_ + cx*this->h_ + cy]/this->h_;
                            box.det_conf = det_conf;
                            box.cls_max_conf = maxConfIndex_[i][cx*this->h_ + cy].conf;
                            box.cls_max_id = maxConfIndex_[i][cx*this->h_ + cy].index;
                            this->boxes_.push_back(box);
//                            std::cout << box.cx << " ls" << std::endl;
                        }
                    }
                }
            }
        }
        return;
    }

protected:
    std::vector<int> anchor_mask_;
    std::vector<int> anchors_;
    int numclass_;
    int numanchors_;
    int stride_;
    int anchor_step_;
    float conf_thresh_;
    int batch_;
    int h_;
    int w_;
    bool is_train = false;
    float ignore_thresh = 0.5;

    std::vector<float>masked_anchor_;

    std::vector<std::vector<MaxConfIndex>> maxConfIndex_;
    std::vector<float>tx_;
    std::vector<float>ty_;
    std::vector<float>tw_;
    std::vector<float>th_;
    std::vector<float>det_conf_;
    std::vector<boxInfo>boxes_;
};

template <typename T,class Context>
class YoloGradientOp final: public Operator<Context> {
public:
 USE_SIMPLE_CTOR_DTOR(YoloGradientOp);
 USE_OPERATOR_CONTEXT_FUNCTIONS;
 void SetDeviceTensor(const std::vector<float>& data, Tensor* tensor) {
     tensor->Resize(data.size());
     context_.template Copy<float, CPUContext, Context>(
     data.size(), data.data(), tensor->template mutable_data<float>());
 }
 void GetTensorToHost(const Tensor* tensor,std::vector<float>& data ) {
     data.resize(tensor->size());
     context_.template Copy<float,Context,CPUContext>(
                 data.size(),tensor->template data<float>(),data.data());

 }

 bool RunOnDevice() override;
};

}//namespace caffe2
#endif // CAFFE2_OPERATORS_RELU_OP_H_
