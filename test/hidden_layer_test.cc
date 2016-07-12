#include <iostream>
#include <iomanip>
#include "../src/matrix.hh"
#include "../src/hidden_layer.hh"

int main() {
  Matrix x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
         y = {{0}, {1}, {1}, {0}};
  HiddenLayer layer_1(2, 4), layer_2(4, 1);
  int idx_epoch, num_epochs = 10000;
  Matrix y_pred_2;

  for (idx_epoch = 0; idx_epoch < num_epochs; ++idx_epoch) {
    Matrix y_pred_1 = layer_1.forward(x);
    // std::cout << "y_pred_1\n" << y_pred_1 << std::endl;
    y_pred_2 = layer_2.forward(y_pred_1);
    // std::cout << "y_pred_2\n" << y_pred_2 << std::endl;
    Matrix error = (y - y_pred_2).cross(y - y_pred_2) / 2.0;
    std::cout << std::setprecision(8) 
              << error.sum() << std::endl;
    Matrix back_2 = layer_2.backward(y - y_pred_2);
    Matrix back_1 = layer_1.backward(back_2);
    layer_1.update(1e-1, 0.8);
    layer_2.update(1e-1, 0.8);
  }
  std::cout << y_pred_2 << std::endl;

  return 0;
}