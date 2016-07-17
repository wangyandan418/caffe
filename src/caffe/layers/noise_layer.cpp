#include <algorithm>
#include <vector>
#include <random>
#include <iostream>

#include "caffe/layers/noise_layer.hpp"

namespace caffe {

template <typename Dtype>
void NoiseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype noise_mean = this->layer_param_.noise_param().mean();
  Dtype noise_std = this->layer_param_.noise_param().std();

  std::default_random_engine generator;
  std::normal_distribution<Dtype> distribution(noise_mean,noise_std);
  for (int i = 0; i < count; ++i) {
	top_data[i] = bottom_data[i] + distribution(generator);
//    top_data[i] = std::max(bottom_data[i], Dtype(0))
//        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
void NoiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
//    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
//    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i];
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(NoiseLayer);
#endif

INSTANTIATE_CLASS(NoiseLayer);
REGISTER_LAYER_CLASS(Noise);
}  // namespace caffe
