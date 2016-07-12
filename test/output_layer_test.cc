#include <iostream>
#include "../src/matrix.hh"
#include "../src/output_layer.hh"

int main() {
  Matrix X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  Matrix y = {{0}, {1}, {1}, {0}};
  OutputLayer layer(2, 1, y);
  std::cout << layer.forward(X) << std::endl;
  std::cout << layer.backward() << std::endl;
  layer.update(1e-1, 8e-1);
  std::cout << layer.output(X) << std::endl;
  return 0;
}