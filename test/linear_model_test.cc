#include <iostream>
#include "../src/linear_model.hh"
#include "../src/layer.hh"
#include "../src/input_layer.hh"
#include "../src/hidden_layer.hh"
#include "../src/output_layer.hh"
#include "../src/matrix.hh"
#include "../src/utils.hh"

int main() {
  Matrix x_train = {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
         y_train = {{0}, {1}, {1}, {0}},
         x_test = {{1, 0}, {0, 0}, {0, 1}, {1, 1}, {0, 0}, {1, 0}};
  int size_hidden_layer_1 = 4, size_hidden_layer_2 = 8;
  nn::param_list params(10000, 4, 1e-1, 8e-2, "logistic", "binary_step");
  InputLayer input_layer;
  HiddenLayer hidden_layer_1(x_train.shape()[1], size_hidden_layer_1);
  HiddenLayer hidden_layer_2(size_hidden_layer_1, size_hidden_layer_2);
  OutputLayer output_layer(size_hidden_layer_2, y_train.shape()[1]);
  LinearModel model;
  model.push(input_layer);
  model.push(hidden_layer_1);
  model.push(hidden_layer_2);
  model.push(output_layer);
  model.compile(params);
  model.input(x_train);
  model.supervise(y_train);
  model.train();
  model.input(x_test);
  std::cout << model.output() << std::endl;
  return 0;
}