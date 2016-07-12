#include <iostream>
#include "../src/perceptron.hh"
#include "../src/matrix.hh"

int main() {
  Matrix x_train = {{1, 0.2, 0.1},
                    {1, 0.3, 0.1},
                    {1, 0.4, 0.7},
                    {1, 0.7, 0.5},
                    {1, 0.3, 0.3},
                    {1, 0.1, 0.3}};
  Matrix y_train = {{0}, {0}, {1}, {1}, {1}, {0}};
  Matrix x_test = {{1, 0.4, 0.6}, {1, 0.1, 0.2}, {1, 0.8, 0.5}};
  Matrix y_test = {{1}, {0}, {1}};
  Perceptron clf(3, 1);
  clf.train(x_train, y_train, 100, 2, 0.01);
  std::cout << clf << std::endl;
  std::cout << clf.predict(x_test) << std::endl;
  return 0; 
}