#include <iostream>
#include <vector>
#include "matrix.hh"
#include "input_layer.hh"

InputLayer::InputLayer(int num_neurons) {
  data_in = Matrix(1, num_neurons);
}
InputLayer::InputLayer(const Matrix & mat_init) {
  data_in = mat_init;
}
std::vector<std::size_t> InputLayer::shape() const {
  std::vector<std::size_t> shape_vec;
  shape_vec.push_back(1);
  shape_vec.push_back(data_in.shape()[1]);
  return shape_vec;
}
void InputLayer::preprocess() {
  // TODO
}
std::istream & operator>>(std::istream & is, InputLayer & layer) {
  if (is) {
  	is >> layer.data_in;
  }
  return is;
}
