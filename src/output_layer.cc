#include <vector>
#include <string>
#include <iostream>
#include <iomanip> 
#include "matrix.hh"
#include "utils.hh"
#include "output_layer.hh"

OutputLayer::OutputLayer(int dim_in, int dim_out, std::string func) {
  weight = Matrix(dim_in, dim_out, "uniform", -1.0, 1.0);
  w_bias = Matrix(1, dim_out, "uniform", -1.0, 1.0);
  activ_func_name = "logistic";
}
OutputLayer::OutputLayer(int dim_in, int dim_out, 
                         const Matrix & output_given_init,
                         std::string func) {
  if (output_given_init.shape()[1] != dim_out) {
    throw "Inconsistent shape";
  }
  weight = Matrix(dim_in, dim_out, "uniform", -1.0, 1.0);
  w_bias = Matrix(1, dim_out, "uniform", -1.0, 1.0);
  output_given = output_given_init;
  activ_func_name = "logistic";
}
OutputLayer::OutputLayer(const Matrix & weight_init, 
                         const Matrix & output_given_init,
                         std::string func) {
  if (weight_init.shape()[1] != output_given_init.shape()[1]) {
    throw "Inconsistent shape";
  }
  weight = weight_init;
  w_bias = Matrix(1, weight.shape()[1], "uniform", -1.0, 1.0);
  output_given = output_given_init;
  activ_func_name = "logistic";
}
OutputLayer::OutputLayer(const Matrix & weight_init, 
                         const Matrix & w_bias_init,
                         const Matrix & output_given_init,
                         std::string func) {
  if (weight_init.shape()[1] != output_given_init.shape()[1] ||
      weight_init.shape()[1] != w_bias_init.shape()[1]) {
    throw "Inconsistent shape";
  }
  weight = weight_init;
  w_bias = w_bias_init;
  output_given = output_given_init;
  activ_func_name = "logistic";
}
Matrix OutputLayer::output() {
  return data_in * weight + w_bias;
}
Matrix OutputLayer::output(const Matrix & mat_in, std::string func) {
  if (weight.shape()[0] == 0) {
    throw "Layer uninitialized";
  }
  if (weight.shape()[0] != mat_in.shape()[1]) {
    throw "Inconsistent shape";
  }
  if (func == "binary_step") {
    return (mat_in * weight + w_bias >= 0.5);
  } else if (func == "identity") {
    return mat_in * weight + w_bias;
  } else {
    throw "Unsupported predict mode";
  }
}
Matrix OutputLayer::forward() {
  return nn::activ_func(data_in * weight + w_bias, activ_func_name);
}
Matrix OutputLayer::forward(int idx_row) {
  return (forward())(idx_row);
} 
Matrix OutputLayer::forward(int idx_r_begin, int idx_r_end) {
  return (forward())(idx_r_begin, idx_r_end);
}
Matrix OutputLayer::forward(const Matrix & data_forward, std::string func) {
  if (weight.shape()[0] == 0) {
    throw "Layer uninitialized";
  }
  if (weight.shape()[0] != data_forward.shape()[1]) {
    throw "Inconsistent shape";
  }
  data_in = data_forward;
  local_field = data_in * weight + w_bias;
  activ_func_name = func;
  return nn::activ_func(local_field, activ_func_name);
}
Matrix OutputLayer::backward() {
  Matrix output_pred = nn::activ_func(local_field, activ_func_name);
  std::cout << std::setprecision(8)
            << ((output_given - output_pred).cross(
                 output_given - output_pred)).sum() 
            << std::endl;
  local_gradient = (output_given - output_pred).cross(
    nn::d_activ_func(local_field, activ_func_name));
  local_field.clear();
  return local_gradient * weight.T();
}
Matrix OutputLayer::backward(int idx_row) {
  Matrix output_pred = nn::activ_func(local_field, activ_func_name);
  std::cout << std::setprecision(8)
            << ((output_given(idx_row) - output_pred).cross(
                 output_given(idx_row) - output_pred)).sum() 
            << std::endl;
  local_gradient = (output_given(idx_row) - output_pred).cross(
    nn::d_activ_func(local_field, activ_func_name));
  local_field.clear();
  return local_gradient * weight.T();  
}
Matrix OutputLayer::backward(int idx_r_begin, int idx_r_end) {
  Matrix output_pred = nn::activ_func(local_field, activ_func_name);
  std::cout << std::setprecision(8)
            << ((output_given(idx_r_begin, idx_r_end) - output_pred).cross(
                 output_given(idx_r_begin, idx_r_end) - output_pred)).sum() 
            << std::endl;
  local_gradient = (output_given(idx_r_begin, idx_r_end) - output_pred).cross(
    nn::d_activ_func(local_field, activ_func_name));
  local_field.clear();
  return local_gradient * weight.T();  
}
Matrix OutputLayer::backward(const Matrix & output_given_temp) {
  Matrix output_pred = nn::activ_func(local_field, activ_func_name);
  local_gradient = (output_given_temp - output_pred).cross(nn::d_activ_func(local_field, activ_func_name));
  local_field.clear();
  return local_gradient * weight.T();
}
void OutputLayer::update(double learning_rate) {
  Matrix bias(data_in.shape()[0], 1, 1.0);
  weight += learning_rate * data_in.T() * local_gradient;
  w_bias += learning_rate * bias.T() * local_gradient;
  data_in.clear();
  local_gradient.clear();
}
void OutputLayer::update(double learning_rate, double momentum) {
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
std::vector<std::size_t> OutputLayer::shape() const {
  return weight.shape();
}
void OutputLayer::supervise(const Matrix & output_given_copy) {
  if (weight.shape()[1] != output_given_copy.shape()[1]) {
    throw "Inconsistent shape";
  }
  // output_given.clear();
  output_given = output_given_copy;
}