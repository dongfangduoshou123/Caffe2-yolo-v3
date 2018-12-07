#include "yolo_op.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/eigen_utils.h"
namespace caffe2{

namespace {
OpSchema::Cost CostInferenceForRelu(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  struct OpSchema::Cost cost = PointwiseCostInference<0>(def, in);
  cost.params_bytes = 0;
  return cost;
}
} // namespace

static int entry_index(int batch, int location, int entry,int w,int h,int outputs,int classes)
{
int n =   location / (w*h);
int loc = location % (w*h);
return batch*outputs + n*w*h*(4+classes+1) + entry*w*h + loc;
}

template <>
bool YoloOp<float,CPUContext>::RunOnDevice() {
auto & X = Input(0);
auto * Y = Output(0);
auto * tx = Output(1);
auto * ty = Output(2);
auto * tw = Output(3);
auto * th = Output(4);
auto * det_conf = Output(5);
auto * class_prob = Output(6);
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
        EigenVectorArrayMap<float> xM(Y->template mutable_data<float>() + index, 2*h*w);
        EigenVectorArrayMap<float>(Y->template mutable_data<float>() + index, 2*h*w) = 1. / (1. + (-xM).exp());
        index = entry_index(b, n*w*h, 4,w,h,h*w*Y->dim(1),this->numclass_);
        EigenVectorArrayMap<float> xM1(Y->template mutable_data<float>() + index, (1+this->numclass_)*w*h);
        EigenVectorArrayMap<float>(Y->template mutable_data<float>() + index, (1+this->numclass_)*w*h) = 1. / (1. + (-xM1).exp());
    }
}

tx->Resize(batch*this->anchor_mask_.size()*h*w);
ty->ResizeLike(*tx);
th->ResizeLike(*tx);
tw->ResizeLike(*tx);
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
        EigenVectorArrayMap<float> xM(Y->template mutable_data<float>() + index, h*w);
        EigenVectorArrayMap<float>(tx->template mutable_data<float>() + step*n, h*w) = xM;

        EigenVectorArrayMap<float> xM1(Y->template mutable_data<float>() + index + h*w, h*w);
        EigenVectorArrayMap<float>(ty->template mutable_data<float>() + step*n, h*w) = xM1;

        EigenVectorArrayMap<float> xM2(Y->template mutable_data<float>() + index + 2*h*w, h*w);
        EigenVectorArrayMap<float>(ty->template mutable_data<float>() + step*n, h*w) = xM2;

        EigenVectorArrayMap<float> xM3(Y->template mutable_data<float>() + index + 3*h*w, h*w);
        EigenVectorArrayMap<float>(ty->template mutable_data<float>() + step*n, h*w) = xM3;

        EigenVectorArrayMap<float> xM4(Y->template mutable_data<float>() + index + 4*h*w, h*w);
        EigenVectorArrayMap<float>(det_conf->template mutable_data<float>() + step*n, h*w) = xM4;

        EigenVectorArrayMap<float> xM5(Y->template mutable_data<float>() + index + 5*h*w, h*w*this->numclass_);
        EigenVectorArrayMap<float>(class_prob->template mutable_data<float>() + step*n*this->numclass_, this->numclass_*h*w) = xM5;
        index += h*w*(5+this->numclass_);
    }
}

std::vector<float>xs,ys;
ys.resize(h*w*batch*this->anchor_mask_.size());
xs.resize(h*w*batch*this->anchor_mask_.size());
for(int i = 0;i < xs.size();i++){
    xs[i] = i%w;
    ys[i] = i/h;
}
std::vector<int>sp;
sp.push_back(batch*this->anchor_mask_.size()*h*w);
tx->Reshape(sp);
EigenVectorArrayMap<float> gridx(xs.data(), h*w*batch*this->anchor_mask_.size());
EigenVectorArrayMap<float> xM7(tx->template mutable_data<float>(), h*w*batch*this->anchor_mask_.size());
EigenVectorArrayMap<float>(tx->template mutable_data<float>(), h*w*batch*this->anchor_mask_.size())
        = xM7 + gridx;

EigenVectorArrayMap<float> gridy(ys.data(), h*w*batch*this->anchor_mask_.size());
EigenVectorArrayMap<float> xM8(ty->template mutable_data<float>(), h*w*batch*this->anchor_mask_.size());
EigenVectorArrayMap<float>(ty->template mutable_data<float>(), h*w*batch*this->anchor_mask_.size())
        = xM8 + gridy;
std::vector<float>aw,ah;
for(int i = 0;i < this->anchor_mask_.size();i++){
    int ind = i*2;
    aw.push_back(this->masked_anchor_[ind]);
    ah.push_back(this->masked_anchor_[ind + 1]);
}
std::vector<float>anchor_w,anchor_h;
for(int i = 0;i < h*w*batch*this->anchor_mask_.size();i ++){
    anchor_w.push_back(aw[i/(h*w*batch)]);
    anchor_h.push_back(ah[i/(h*w*batch)]);
}
EigenVectorArrayMap<float> anchorw(anchor_w.data(), h*w*batch*this->anchor_mask_.size());
EigenVectorArrayMap<float> xM9(tw->template mutable_data<float>(), h*w*batch*this->anchor_mask_.size());
EigenVectorArrayMap<float>(tw->template mutable_data<float>(), h*w*batch*this->anchor_mask_.size())
    = anchorw*(xM9.exp());

EigenVectorArrayMap<float> anchorh(anchor_h.data(), h*w*batch*this->anchor_mask_.size());
EigenVectorArrayMap<float> xM10(th->template mutable_data<float>(), h*w*batch*this->anchor_mask_.size());
EigenVectorArrayMap<float>(th->template mutable_data<float>(), h*w*batch*this->anchor_mask_.size())
    = anchorh*(xM10.exp());

//    std::vector<int>psp;
//    psp.push_back(batch*this->anchor_mask_.size());
//    psp.push_back(this->numclass);
//    psp.push_back(h*w);
//    class_prob->Reshape(psp);
//    long step = this->numclass_*h*w;
//    for(int i = 0;i < batch*this->anchor_mask_.size();i++){
//        Eigen::Matrix::Index maxRow;
//        auto max = Eigen::Map<Eigen::Matrix<float, this->numclass_, h*w> >(class_prob->template mutable_data<float>() + i*step).colwise().maxCoeff();
//    }


return true;
}

template <>
bool YoloGradientOp<float,CPUContext>::RunOnDevice() {
CAFFE_NOT_IMPLEMENTED;
}

REGISTER_CPU_OPERATOR(Yolo,YoloOp<float,CPUContext>);
REGISTER_CPU_OPERATOR(YoloGradient,YoloGradientOp<float,CPUContext>);

OPERATOR_SCHEMA(Yolo)
.NumInputs(1,2)
.NumOutputs(8)
.AllowInplace({{0, 0}})
.CostInferenceFunction(CostInferenceForRelu)
.IdenticalTypeAndShape()
.SetDoc(R"DOC(
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
)DOC")
.Input(0, "X", "1D input tensor")
.Output(0, "Y", "1D input tensor");

OPERATOR_SCHEMA(YoloGradient).NumOutputs(1);

class GetYoloGradient : public GradientMakerBase {
using GradientMakerBase::GradientMakerBase;
vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
                def_.type() + "Gradient",
                "",
                vector<string>{I(0),I(1),O(0),GO(0)},
                vector<string>{GI(0)});
}
};

REGISTER_GRADIENT(Yolo, GetYoloGradient);

} //namespace caffe2
