#include <iostream>
#include "../src/matrix.hh"
#include "../src/layer.hh"

int main() {
  Matrix m = {{1, 2, 3}, {4, 5, 6}}, 
         n = {{1.2, 2.3}, {3.4, 4.5}, {5.6, 6.7}};
  Layer layer;
  layer.input(m);
  std::cout << layer.output() << std::endl;
  std::cout << layer.output(n) << std::endl;
  std::cout << layer.forward() << std::endl;
  std::cout << layer.forward(1) << std::endl;
  std::cout << layer.forward(1, 2) << std::endl;
  std::cout << layer.forward(n) << std::endl;
  std::cout << layer.backward() << std::endl;
  std::cout << layer.backward(n) << std::endl;
  layer.update(1e-1);
  layer.update(1e-1, 8e-1);
  std::cout << layer.shape()[0] << ' ' << layer.shape()[1] << std::endl;
  return 0;
}