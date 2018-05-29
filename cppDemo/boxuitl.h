#ifndef BOXUITL_H
#define BOXUITL_H
#include <iostream>
#include <caffe2/core/tensor.h>
#include <caffe2/core/context_gpu.h>
#include <caffe2/core/predictor.h>
#include <strstream>
#include <iosfwd>
using namespace caffe2;
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

template<typename T>
std::vector<int> argsort(const vector<T>& a){
    int Len = a.size();
    std::vector<int>idx(Len,0);
    for(int i = 0; i < Len; i++){
        idx[i] = i;
    }
    std::sort(idx.begin(), idx.end(), [&a](int i1, int i2){return a[i1]< a[i2];});
    return idx;
}
float bbox_iou(const boxInfo& box1,const boxInfo& box2,bool x1y1x2y2 = true){
    float mx,Mx , my , My , w1 ,h1 , w2 ,h2 ;
    if(x1y1x2y2){
        mx = std::min(box1.cx,box2.cx);
        Mx = std::max(box1.w,box2.w);
        my = std::min(box1.cy,box2.cy);
        My = std::max(box1.h,box2.h);
        w1 = box1.w - box1.cx;
        h1 = box1.h - box1.cy;
        w2 = box2.w - box2.cx;
        h2 = box2.h - box2.cy;
    }else{
        mx = std::min(box1.cx-box1.w/2.0, box2.cx-box2.w/2.0);
            Mx = std::max(box1.cx+box1.w/2.0, box2.cx+box2.w/2.0);
               my = std::min(box1.cy-box1.h/2.0, box2.cy-box2.h/2.0);
               My = std::max(box1.cy+box1.h/2.0, box2.cy+box2.h/2.0);
               w1 = box1.w;
               h1 = box1.h;
               w2 = box2.w;
                h2 = box2.h;
    }

    float uw = Mx - mx;
        float uh = My - my;
       float cw = w1 + w2 - uw;
       float ch = h1 + h2 - uh;
       float carea = 0;
        if (cw <= 0 or ch <= 0)
            return 0.0;

       float area1 = w1 * h1;
       float area2 = w2 * h2;
       carea = cw * ch;
      float  uarea = area1 + area2 - carea;
        return carea/uarea;
}

class YOLOBoxParser{
public:
    std::vector<boxInfo> nms(std::vector<boxInfo>& inboxes,float nms_thresh){
        std::vector<boxInfo>outboxes;
        outboxes.resize(0);
//        for(int i = 0;i < inboxes.size();i ++){
//            std::cout << inboxes[i].cx << " " << inboxes[i].cy << " "<< inboxes[i].det_conf << " " << inboxes[i].cls_max_conf << std::endl;
//        }
        if(inboxes.size() == 0)
            return outboxes;
        std::vector<float>def_confs;
        def_confs.resize(inboxes.size());
        for(int i = 0;i < inboxes.size();i ++){
            def_confs[i] = 1-boxes_[i].det_conf;
        }
        std::vector<int>sortIds = argsort<float>(def_confs);
        for(int i = 0;i < inboxes.size();i ++){
            boxInfo box_i = inboxes[sortIds[i]];
            if (box_i.det_conf > 0){
                outboxes.push_back(box_i);
                for( int j = i + 1;j < inboxes.size(); j ++){
                    boxInfo box_j = inboxes[sortIds[j]];
                    if(bbox_iou(box_i,box_j,false) >nms_thresh){
                        inboxes[sortIds[j]].det_conf = 0;
                    }
                }
            }
        }
        return outboxes;
    }

    std::vector<boxInfo> parerToHost(caffe2::Workspace& workspace){
        boxes_.resize(0);
        for(int i= 1;i < 4;i ++){
            std::vector<float>tx;
            std::vector<float>ty;
            std::vector<float>tw;
            std::vector<float>th;
            std::vector<float>det_conf;
            std::vector<float>class_prob;
            std::stringstream ss;
            ss << "gpu_0/tx_" << i ;
            auto tx_tensor = workspace.GetBlob(ss.str())->template GetMutable<caffe2::Tensor<CUDAContext>>();
            this->GetTensorToHost(workspace.GetBlob(ss.str())->template GetMutable<caffe2::Tensor<CUDAContext>>(),tx);
            ss.str("");
            ss << "gpu_0/ty_" << i ;
            this->GetTensorToHost(workspace.GetBlob(ss.str())->template GetMutable<caffe2::Tensor<CUDAContext>>(),ty);
            ss.str("");
            ss << "gpu_0/tw_" << i ;
            this->GetTensorToHost(workspace.GetBlob(ss.str())->template GetMutable<caffe2::Tensor<CUDAContext>>(),tw);
            ss.str("");
            ss << "gpu_0/th_" << i ;
            this->GetTensorToHost(workspace.GetBlob(ss.str())->template GetMutable<caffe2::Tensor<CUDAContext>>(),th);
            ss.str("");
            ss << "gpu_0/det_conf_" << i ;
            this->GetTensorToHost(workspace.GetBlob(ss.str())->template GetMutable<caffe2::Tensor<CUDAContext>>(),det_conf);
            ss.str("");
            ss << "gpu_0/class_prob_" << i ;
            auto class_tensor = workspace.GetBlob(ss.str())->template GetMutable<caffe2::Tensor<CUDAContext>>();
            this->GetTensorToHost(workspace.GetBlob(ss.str())->template GetMutable<caffe2::Tensor<CUDAContext>>(),class_prob);
            ss.str("");
            computeMaxConfIndex(class_tensor,80);
            this->GetYoloBoxes(tx_tensor->dim(0),3,tx_tensor->dim(3),tx_tensor->dim(2),det_conf,tx,ty,tw,th);
        }
        std::vector<boxInfo> outboxes = this->nms(this->boxes_,0.4);
        return outboxes;
    }

    void GetTensorToHost(const Tensor<CUDAContext>* tensor,std::vector<float>& data ) {
        data.resize(tensor->size());
        CUDAContext context_;
        context_.template Copy<float,CUDAContext,CPUContext>(
                    data.size(),tensor->template data<float>(),data.data());

    }

    void computeMaxConfIndex(Tensor<CUDAContext>* tensor,int numclass){
        std::vector<float>data;
        GetTensorToHost(tensor,data);
        maxConfIndex_.resize(tensor->dim(0));
        for(int i = 0;i < maxConfIndex_.size();i++){
            std::vector<MaxConfIndex> cur;
            cur.resize(tensor->dim(2));
            int offset = tensor->dim(2)*numclass;
            for(int j = 0;j < numclass;j ++){

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

    void GetYoloBoxes(int batch,int anchors,int w,int h,std::vector<float> det_conf_,std::vector<float>tx,std::vector<float> ty,
                      std::vector<float> tw,std::vector<float> th){
        for(int b = 0;b < batch;b ++){
            for(int i = 0;i < anchors;i ++){
                for(int cx = 0;cx < w;cx++){
                    for(int cy = 0;cy < h;cy ++){
                        float det_conf = det_conf_[i*w*h + cx*h + cy];
                        float conf = det_conf;
                        float conf_thresh = 0.6;
                        if(conf > conf_thresh){
                            boxInfo box;
                            box.cx = tx[i*w*h + cx*h + cy]/w;
                            box.cy = ty[i*w*h + cx*h + cy]/h;
                            box.w = tw[i*w*h + cx*h + cy]/w;
                            box.h = th[i*w*h + cx*h + cy]/h;
                            box.det_conf = det_conf;
                            box.cls_max_conf = maxConfIndex_[i][cx*h + cy].conf;
                            box.cls_max_id = maxConfIndex_[i][cx*h + cy].index;
                            boxes_.push_back(box);
                        }
                    }
                }
            }
        }
        return;
    }
private:
    std::vector<std::vector<MaxConfIndex>> maxConfIndex_;
    std::vector<boxInfo> boxes_;
};

#endif // BOXUITL_H
