#include "caffe2/core/context_gpu.h"
#include "yolo_op.h"
#include "caffe2/utils/math.h"
#include <stdio.h>

namespace caffe2 {
namespace {
typedef struct{
    float x, y, w, h;
} box;

int int_index(int *a, int val, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        if(a[i] == val) return i;
    }
    return -1;
}


box float_to_box(const float *f, int stride)
{
    box b = {0};
    b.x = f[0];
    b.y = f[1*stride];
    b.w = f[2*stride];
    b.h = f[3*stride];
    return b;
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

box get_yolo_box(const float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

float delta_yolo_box(box truth, const float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*lw - i);
    float ty = (truth.y*lh - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);
//    LOG(INFO) << tx << " " << ty << " " << tw << " " << th << " " << i << " " << j << " truthbox";
//    LOG(INFO) << x[index + 0*stride]<< " " << x[index + 1*stride]<< " " << x[index + 2*stride] << " " << x[index + 3*stride] << " predbox, scale:" << scale;
    delta[index + 0*stride] = scale * (tx - x[index + 0*stride])*(-1);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride])*(-1);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride])*(-1);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride])*(-1);
    return iou;
}

void delta_yolo_class(const float *output, float *delta, int index, int classs, int classes, int stride, float *avg_cat)
{
    int n;
    if (delta[index]){
        delta[index + stride*classs] = (-1)*(1 - output[index + stride*classs]);
//        LOG(INFO) << delta[index + stride*classs];
        if(avg_cat) *avg_cat += output[index + stride*classs];
        return;
    }
    for(n = 0; n < classes; ++n){
        delta[index + stride*n] =(-1)*( ((n == classs)?1 : 0) - output[index + stride*n]);
        if(n == classs && avg_cat) *avg_cat += output[index + stride*n];
    }
}

static int entry_index(int batch, int location, int entry,int w,int h,int outputs,int classes)
{
    int n =   location / (w*h);
    int loc = location % (w*h);
    return batch*outputs + n*w*h*(4+classes+1) + entry*w*h + loc;
}

template <typename T>
__global__ void fillKernel(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = x[i];
  }
}


template <typename T>
__global__ void SigmoidKernel(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = 1. / (1. + exp(-x[i]));
  }
}

template <typename T>
__global__ void GenerateGridXKernel(const int N, const int w,const int h,T* out1/*,T* out2*/) {
//    int index = 0;
  CUDA_1D_KERNEL_LOOP(i, N) {
    out1[i] = out1[i] + i%w;
  }
}

template <typename T>
__global__ void GenerateGridYKernel(const int N, const int w,const int h,T* out1/*,T* out2*/) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    out1[i] = out1[i] + int((i+0.0001)/h);
  }
}

template <typename T>
__global__ void twKernel(const int N,const int batch,const int w,const int h, T* x,T* y1) {
  CUDA_1D_KERNEL_LOOP(i, N) {
          y1[i] = exp(y1[i])*x[i/(batch*h*w)];
//          y2[i] = exp(y2[i])*x[i/(batch*h*w) + offset];
  }
}

template <typename T>
__global__ void thKernel(const int N,const int batch,const int w,const int h, T* x,T* y2) {
  CUDA_1D_KERNEL_LOOP(i, N) {
          y2[i] = exp(y2[i])*x[i/(batch*h*w)];
  }
}

}  // namespace


template<>
bool YoloOp<float,CUDAContext>::RunOnDevice() {
    auto& X = Input(0);
    auto * Y = Output(0);
    auto * tx = Output(1);
    auto * ty = Output(2);
    auto * tw = Output(3);
    auto * th = Output(4);
    auto * det_conf = Output(5);
    auto * class_prob = Output(6);
    auto * tmp_blob = Output(7);
    Y->ResizeLike(X);
    Y->CopyFrom(X);

    int batch = Y->dim(0);
    int h = Y->dim(2);
    int w = Y->dim(3);
    this->h_=h;
    this->w_=w;
    this->batch_ = batch;
    for (int b = 0; b < batch; ++b){
        for(int n = 0; n < this->anchor_mask_.size(); ++n){
            int index = entry_index(b, n*w*h, 0,w,h,h*w*Y->dim(1),this->numclass_);
            SigmoidKernel<float><<<CAFFE_GET_BLOCKS(2*h*w), CAFFE_CUDA_NUM_THREADS,
                            0,context_.cuda_stream()>>>(2*h*w,Y->template mutable_data<float>() + index ,
                                                                Y->template mutable_data<float>() + index);
            index = entry_index(b, n*w*h, 4,w,h,h*w*Y->dim(1),this->numclass_);
            SigmoidKernel<float><<<CAFFE_GET_BLOCKS((1+this->numclass_)*w*h), CAFFE_CUDA_NUM_THREADS,
                            0,context_.cuda_stream()>>>((1+this->numclass_)*w*h, Y->template mutable_data<float>() + index
                                                                , Y->template mutable_data<float>() + index);
        }
    }
    tx->Resize(batch*this->anchor_mask_.size()*h*w);
    ty->ResizeLike(*tx);
    th->ResizeLike(*tx);
    tw->ResizeLike(*tx);
    tmp_blob->Resize(batch*this->anchor_mask_.size()*h*w);
    det_conf->ResizeLike(*tx);

    std::vector<int>shape;
    shape.push_back(batch);
    shape.push_back(this->anchor_mask_.size());
    shape.push_back(h);
    shape.push_back(w);
    tx->Reshape(shape);
    int index = 0;
    int step = h*w;
    class_prob->Resize(batch*this->anchor_mask_.size()*h*w*this->numclass_);

    for(int b = 0; b < batch; ++b){
        for(int n = 0; n < this->anchor_mask_.size(); ++n){
            fillKernel<float><<<CAFFE_GET_BLOCKS(h*w), CAFFE_CUDA_NUM_THREADS,
                            0,context_.cuda_stream()>>>(h*w,Y->template mutable_data<float>() + index ,
                                                                tx->template mutable_data<float>() + step*n);

            fillKernel<float><<<CAFFE_GET_BLOCKS(h*w), CAFFE_CUDA_NUM_THREADS,
                            0,context_.cuda_stream()>>>(h*w,Y->template mutable_data<float>() + index + h*w ,
                                                                ty->template mutable_data<float>() + step*n);

            fillKernel<float><<<CAFFE_GET_BLOCKS(h*w), CAFFE_CUDA_NUM_THREADS,
                            0,context_.cuda_stream()>>>(h*w,Y->template mutable_data<float>() + index + 2*h*w ,
                                                                tw->template mutable_data<float>() + step*n);

            fillKernel<float><<<CAFFE_GET_BLOCKS(h*w), CAFFE_CUDA_NUM_THREADS,
                            0,context_.cuda_stream()>>>(h*w,Y->template mutable_data<float>() + index + 3*h*w ,
                                                                th->template mutable_data<float>() + step*n);

            fillKernel<float><<<CAFFE_GET_BLOCKS(h*w), CAFFE_CUDA_NUM_THREADS,
                            0,context_.cuda_stream()>>>(h*w,Y->template mutable_data<float>() + index + 4*h*w ,
                                                                det_conf->template mutable_data<float>() + step*n);

            fillKernel<float><<<CAFFE_GET_BLOCKS(h*w*this->numclass_), CAFFE_CUDA_NUM_THREADS,
                            0,context_.cuda_stream()>>>(h*w*this->numclass_,Y->template mutable_data<float>() + index + 5*h*w ,
                                                                class_prob->template mutable_data<float>() + step*n*this->numclass_);
            index += h*w*(5+this->numclass_);
        }
    }

    std::vector<int>sp;
    sp.push_back(batch*this->anchor_mask_.size()*h*w);
    tx->Reshape(sp);
    GenerateGridXKernel<float><<<CAFFE_GET_BLOCKS(h*w*batch*this->anchor_mask_.size()),CAFFE_CUDA_NUM_THREADS,
            0,context_.cuda_stream()>>>(h*w*this->anchor_mask_.size()*batch,w,h,
                                        tx->template mutable_data<float>());

    for(int i = 0;i <batch*this->anchor_mask_.size();i++){
        GenerateGridYKernel<float><<<CAFFE_GET_BLOCKS(h*w),CAFFE_CUDA_NUM_THREADS,
                0,context_.cuda_stream()>>>(h*w,w,h,
                                            ty->template mutable_data<float>() + i*h*w);
    }


    std::vector<float>aw,ah;
    for(int i = 0;i < this->anchor_mask_.size();i++){
        int ind = i*2;
        aw.push_back(this->masked_anchor_[ind]);
        ah.push_back(this->masked_anchor_[ind + 1]);
    }
    this->SetDeviceTensor(aw,tmp_blob);
    twKernel<float><<<CAFFE_GET_BLOCKS(h*w*batch*this->anchor_mask_.size()),CAFFE_CUDA_NUM_THREADS,
            0,context_.cuda_stream()>>>(h*w*this->anchor_mask_.size()*batch,batch,w,h,tmp_blob->template mutable_data<float>(),
                                        tw->template mutable_data<float>());

    this->SetDeviceTensor(ah,tmp_blob);
    thKernel<float><<<CAFFE_GET_BLOCKS(h*w*batch*this->anchor_mask_.size()),CAFFE_CUDA_NUM_THREADS,
            0,context_.cuda_stream()>>>(h*w*this->anchor_mask_.size()*batch,batch,w,h,tmp_blob->template mutable_data<float>(),
                                        th->template mutable_data<float>());
    std::vector<int>psp;
    psp.push_back(batch*this->anchor_mask_.size());
    psp.push_back(this->numclass_);
    psp.push_back(h*w);
    class_prob->Reshape(psp);
    tx->Reshape(shape);

    if(!is_train)return true;
    ////////////////////compute yolo loss/////////////////////
    int imghw = 416;
    std::vector<float>acdata;
    for(auto& it:this->anchors_){
        acdata.push_back(it);
    }
    auto& nettruth = Input(1);
    std::vector<float>X_host;
    GetTensorToHost(Y,X_host);
    math::Set<float,caffe2::CUDAContext>(Y->size(),0,Y->template mutable_data<float>(),&context_);
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;

    std::vector<float>nettruth_host;
    std::vector<float>Y_host;

    GetTensorToHost(&nettruth,nettruth_host);
    GetTensorToHost(Y,Y_host);
    for(int b = 0;b < X.dim(0); ++b){
        for(int j = 0;j < X.dim(2); ++j){
            for(int i = 0; i < X.dim(3); ++i){
                for(int n = 0;n < this->anchor_mask_.size(); ++n){
                    int box_index = entry_index(b, n*w*h + j*w + i, 0,w,h,h*w*X.dim(1),this->numclass_);
                    box pred = get_yolo_box((const float*)X_host.data(), (float*)(acdata.data()),
                                            anchor_mask_[n], box_index, i, j, X.dim(3), X.dim(2), imghw, imghw, X.dim(3)*X.dim(2));
                    float best_iou = 0;
                    int best_t = 0;
                    int max_boxes = 90;
                    for(int t = 0; t < max_boxes; ++t){
                        box truth = float_to_box((float*)nettruth_host.data() + t*(4 + 1) + b*max_boxes*5 + 1, 1);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    int obj_index = entry_index(b, n*w*h + j*w + i, 4,w,h,h*w*X.dim(1),this->numclass_);
                    avg_anyobj += X_host[obj_index];
                    Y_host[obj_index] = -1*(0- X_host[obj_index]);
                    if (best_iou > this->ignore_thresh) {
                        Y_host[obj_index] = 0;
//                        LOG(INFO) << Y_host[obj_index];
                    }
//                    LOG(INFO) << best_iou << " " << this->ignore_thresh << " " << Y_host[obj_index] << " " << X_host[obj_index];
//                    if (best_iou > l.truth_thresh) {
//                        Y_host[obj_index] = 1 - X.template mutable_data<float>()[obj_index];

//                        int class = net.truth[best_t*(4 + 1) + b*l.truths + 4];
//                        if (l.map) class = l.map[class];
//                        int class_index = entry_index( b, n*l.w*l.h + j*l.w + i, 4 + 1,w,h,h*w*X.dim(1),this->numclass_);
//                        delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, 0);
//                        box truth = float_to_box(net.truth + best_t*(4 + 1) + b*l.truths, 1);
//                        delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);

//                    }
                }
            }
        }
        int max_boxes = nettruth.dim(1);
        for(int t = 0; t < max_boxes; ++t){

            box truth = float_to_box((float*)nettruth_host.data() + t*(4 + 1) + b*max_boxes*5+1, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            int i = (truth.x * w);
            int j = (truth.y * h);
            box truth_shift = truth;
            box bestbox;
            truth_shift.x = truth_shift.y = 0;
            for(int n = 0; n < this->anchors_.size()/2; ++n){
                box pred = {0};
                pred.x = 0;
                pred.y = 0;
                pred.w = (this->anchors_[2*n]+0.00001)/imghw;
                pred.h = (this->anchors_[2*n+1]+0.00001)/imghw;
                float iou = box_iou(pred, truth_shift);
//                LOG(INFO) << iou;
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                    bestbox = pred;
                }
            }
            int mask_n = int_index((int*)(this->anchor_mask_.data()), best_n, this->anchor_mask_.size());
            if(mask_n >= 0){
//                LOG(INFO) << bestbox.x << " " << bestbox.y << " " << bestbox.w << " " << bestbox.h << " " << this->anchors_[0] << " " << imghw;
//                LOG(INFO) << "dim:" << w << " mask_n:" << mask_n;
                int box_index = entry_index(b, mask_n*w*h + j*w + i, 0,w,h,h*w*X.dim(1),this->numclass_);
                float iou = 0;
                iou = delta_yolo_box(truth,(float*)X_host.data(),(float*)acdata.data(), best_n, box_index, i, j, w, h,imghw, imghw, (float*)Y_host.data(), (2-truth.w*truth.h), w*h);
                int obj_index = entry_index(b, mask_n*w*h + j*w + i, 4, w,h,h*w*X.dim(1),this->numclass_);
                avg_obj += X_host[obj_index];
//                LOG(INFO) << "obj conf:" << X_host[obj_index] << " " << w;
                Y_host[obj_index] =-1*(1 - X_host[obj_index]);
//                LOG(INFO) << Y_host[obj_index];
                int classs = nettruth_host[t*(4 + 1) + b*max_boxes*5];
//                LOG(INFO) << "classid:" << classs;
                int class_index = entry_index(b, mask_n*w*h + j*w + i, 4 + 1,w,h,h*w*X.dim(1),this->numclass_);
                delta_yolo_class((float*)X_host.data(), (float*)Y_host.data(), class_index, classs,this->numclass_, w*h, &avg_cat);

                ++count;
                ++class_count;
                if(iou > .5) recall += 1;
                if(iou > .75) recall75 += 1;
                avg_iou += iou;
            }
        }
    }

////    cost = pow(mag_array(Y_host, X.size() * batch), 2);
    if(count == 0){
//        LOG(INFO) << "No Box match This Yolo Layer's anchors.";
    }else{
        LOG(INFO) << "Avg IOU:" << avg_iou/count
              << ", Class:" << avg_cat/class_count
              << ", Obj:" << avg_obj/count
              << ", No Obj:" << avg_anyobj/(w*h*this->anchor_mask_.size()*batch)
              << ", .5R:" << recall/count
              << ", .75R:" << recall75/count
              << ", count:" << count ;
    }
    SetDeviceTensor(Y_host,Y);
    shape.clear();
    shape.push_back(X.dim(0));
    shape.push_back(X.dim(1));
    shape.push_back(X.dim(2));
    shape.push_back(X.dim(3));
    Y->Reshape(shape);
    return true;
}

template<>
bool YoloGradientOp<float,CUDAContext>::RunOnDevice() {
    auto& X = Input(0);
    auto *dX = Output(0);
    auto& dY = Input(2);
    dX->ResizeLike(dY);
//    caffe2::math::Axpy<float,CUDAContext>(
//                dX->size(),1.0,dY.template data<float>(),
//                dX->template mutable_data<float>(),&context_);
    dX->CopyFrom(dY);
//    TensorCPU* Ycpu1 = new caffe2::TensorCPU();
//    Ycpu1->CopyFrom(*dX);
//    for(int i = 0;i < 100;i ++){
//        LOG(INFO) << Ycpu1->template data<float>()[i];
//    }
//    LOG(INFO) << Ycpu1->dim(2) << " dim";
//    std::string blobname1 = "/opt/yolo1.blob";
//    if(dY.dim(3) == 13)
//        blobname1 = "/opt/yolo1-input.blob";
//    else if(dY.dim(3) == 26)
//        blobname1 = "/opt/yolo2-input.blob";
//    else
//        blobname1 = "/opt/yolo3-input.blob";
//    TensorPrinter tpter2("data",blobname1,100000000000);
//    tpter2.Print<float>(*Ycpu1);

//    math::Set<float, CUDAContext>(
//        dX->size(),
//        0.0,
//        dX->template mutable_data<float>(),
//        &context_);
    return true;
}

REGISTER_CUDA_OPERATOR(Yolo,YoloOp<float,CUDAContext>);
REGISTER_CUDA_OPERATOR(YoloGradient,YoloGradientOp<float,CUDAContext>);
}  // namespace caffe2

