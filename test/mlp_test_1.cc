#include <iostream>
#include <iomanip>
#include "../src/matrix.hh"
#include "../src/input_layer.hh"
#include "../src/hidden_layer.hh"
#include "../src/output_layer.hh"

int main() {
  Matrix X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
         Y = {{0}, {1}, {1}, {0}};
  int size_hidden_layer = 4,
      num_epochs = 50000, idx_epoch;
  InputLayer input_layer(X);
  HiddenLayer hidden_layer(X.shape()[1], size_hidden_layer);
  OutputLayer output_layer(size_hidden_layer, Y.shape()[1], Y);
  for (idx_epoch = 0; idx_epoch < num_epochs; ++idx_epoch) {
    Matrix Y_pred = output_layer.forward(hidden_layer.forward(input_layer.forward()));
    std::cout << idx_epoch<< " "; // << std::setprecision(8) 
              // << ((Y - Y_pred).cross(Y - Y_pred)).sum() << std::endl;
    hidden_layer.backward(output_layer.backward());
    hidden_layer.update(1e-1);
    output_layer.update(1e-1);
  }
  std::cout << output_layer.output(hidden_layer.forward(input_layer.forward()))
            << std::endl;
  return 0;
}