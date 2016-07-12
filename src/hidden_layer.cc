#include <vector>
#include <string>
#include "matrix.hh"
#include "utils.hh"
#include "hidden_layer.hh"

HiddenLayer::HiddenLayer(int dim_in, int num_neurons, std::string func) {
  weight = Matrix(dim_in, num_neurons, "uniform", -1.0, 1.0);
  w_bias = Matrix(1, num_neurons, "uniform", -1.0, 1.0);
  activ_func_name = func;
}
HiddenLayer::HiddenLayer(const Matrix & weight_init, std::string func) {
  weight = weight_init;
  w_bias = Matrix(1, weight.shape()[1], "uniform", -1.0, 1.0);
  activ_func_name = func;
}
HiddenLayer::HiddenLayer(const Matrix & weight_init, 
                         const Matrix & w_bias_init,
                         std::string func) {
  if (weight_init.shape()[1] != w_bias_init.shape()[1]) {
    throw "Inconsistent shape";
  }
  weight = weight_init;
  w_bias = w_bias_init;
  activ_func_name = func;
}
Matrix HiddenLayer::output() {
  return data_in * weight + w_bias;
}
Matrix HiddenLayer::output(const Matrix & mat_in) {
  return mat_in * weight + w_bias;
}
Matrix HiddenLayer::forward() {
  return nn::activ_func(data_in * weight + w_bias, activ_func_name);
}
Matrix HiddenLayer::forward(int idx_row) {
  return (forward())(idx_row);
}
Matrix HiddenLayer::forward(int idx_r_begin, int idx_r_end) {
  return (forward())(idx_r_begin, idx_r_end);
}
Matrix HiddenLayer::forward(const Matrix & data_forward, std::string func) {
  if (weight.shape()[0] == 0) {
    throw "Layer uninitialized";
  }
  if (weight.shape()[0] != data_forward.shape()[1]) {
    throw "Inconsistent shape";
  }
  activ_func_name = func;
  data_in = data_forward;
  local_field = data_in * weight + w_bias;
  return nn::activ_func(local_field, activ_func_name);
}
Matrix HiddenLayer::backward() {
  return Matrix(0, 0);
}
Matrix HiddenLayer::backward(int idx_row) {
  return (backward())(idx_row);
}
Matrix HiddenLayer::backward(int idx_r_begin, int idx_r_end) {
  return (backward())(idx_r_begin, idx_r_end);
}
Matrix HiddenLayer::backward(const Matrix & delta_backward) {
  local_gradient = nn::d_activ_func(local_field, activ_func_name).cross(delta_backward);
  local_field.clear();
  return local_gradient * weight.T();
}
void HiddenLayer::update(double learning_rate) {
  Matrix bias(data_in.shape()[0], 1, 1.0);
  weight += learning_rate * data_in.T() * local_gradient;
  w_bias += learning_rate * bias.T() * local_gradient;
  data_in.clear();
  local_gradient.clear();
}
void HiddenLayer::update(double learning_rate, double momentum) {
  if (delta_weight.shape()[0] == 0) {
    delta_weight = Matrix(weight.shape()[0], weight.shape()[1]);
  }
  if (delta_w_bias.shape()[0] == 0) {
    delta_w_bias = Matrix(w_bias.shape()[0], w_bias.shape()[1]);
  }
  Matrix bias(data_in.shape()[0], 1, 1.0);
  delta_weight = momentum * delta_weight + 
                 learning_rate * data_in.T() * local_gradient;
  delta_w_bias = momentum * delta_w_bias + 
                 learning_rate * bias.T() * local_gradient;
  weight += delta_weight;
  w_bias += delta_w_bias;
  data_in.clear();
  local_gradient.clear();    
}
std::vector<size_t> HiddenLayer::shape() const {
  return weight.shape();
}
