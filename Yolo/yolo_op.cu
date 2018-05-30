#include "caffe2/core/context_gpu.h"
#include "yolo_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace {

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
//    out2[i] = (out2[i] + index/h);
//    index++;
//    if(index == h*w)
//        index = 0;
  }
}

template <typename T>
__global__ void GenerateGridYKernel(const int N, const int w,const int h,T* out1/*,T* out2*/) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    out1[i] = out1[i] + int((i+0.0001)/h);
  }
}

template <typename T>
__global__ void thtwKernel(const int N,const int batch,const int w,const int h, T* x,T* y1,T* y2,int offset) {
  CUDA_1D_KERNEL_LOOP(i, N) {
          y1[i] = exp(y1[i])*x[i/(batch*h*w)];
          y2[i] = exp(y2[i])*x[i/(batch*h*w) + offset];
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
    this->SetDeviceTensor(ah,tmp_blob,this->anchor_mask_.size());
    thtwKernel<float><<<CAFFE_GET_BLOCKS(h*w*batch*this->anchor_mask_.size()),CAFFE_CUDA_NUM_THREADS,
            0,context_.cuda_stream()>>>(h*w*this->anchor_mask_.size()*batch,batch,w,h,tmp_blob->template mutable_data<float>(),
                                        tw->template mutable_data<float>(),th->template mutable_data<float>(),this->anchor_mask_.size());

//    thtwKernel<float><<<CAFFE_GET_BLOCKS(h*w*batch*this->anchor_mask_.size()),CAFFE_CUDA_NUM_THREADS,
//            0,context_.cuda_stream()>>>(h*w*this->anchor_mask_.size()*batch,batch,w,h,tmp_blob->template mutable_data<float>(),th->template mutable_data<float>());

//    std::vector<int>psp;
//    psp.push_back(batch*this->anchor_mask_.size());
//    psp.push_back(this->numclass_);
//    psp.push_back(h*w);
//    class_prob->Reshape(psp);
//    computeMaxConfIndex(class_prob);
//    GetTensorToHost(tx,this->tx_);
//    GetTensorToHost(ty,this->ty_);
//    GetTensorToHost(th,this->th_);
//    GetTensorToHost(tw,this->tw_);
//    GetTensorToHost(det_conf,this->det_conf_);
//    GetYoloBoxes();
    tx->Reshape(shape);
    return true;
}

template<>
bool YoloGradientOp<float,CUDAContext>::RunOnDevice() {
    CAFFE_NOT_IMPLEMENTED;
}

REGISTER_CUDA_OPERATOR(Yolo,YoloOp<float,CUDAContext>);
REGISTER_CUDA_OPERATOR(YoloGradient,YoloGradientOp<float,CUDAContext>);
}  // namespace caffe2

