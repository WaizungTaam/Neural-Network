#include <iostream>
#include "../src/matrix.hh"
#include "../src/input_layer.hh"

int main() {
  Matrix m = {{1.1, 2.3}, {3.6, 2.7}, {2.5, 6.7}},
         n = {{1, 2}, {3, 4}};
  InputLayer layer_1(m);
  InputLayer layer_2 = layer_1;
  InputLayer layer_3(2);
  layer_1.input(n);
  std::cout << layer_1.output() << std::endl;
  std::cout << layer_1.output(m) << std::endl;
  std::cout << layer_1.forward() << std::endl;
  std::cout << layer_1.forward(0) << std::endl;
  std::cout << layer_1.forward(0, 1) << std::endl;
  std::cout << layer_1.forward(m) << std::endl;
  std::cout << layer_1.backward() << std::endl;
  std::cout << layer_1.backward(m) << std::endl;
  layer_1.update(1e-1);
  layer_1.update(1e-1, 8e-1);
  std::cout << layer_1.shape()[0] << " " << layer_1.shape()[1] << std::endl;
  return 0; 
}