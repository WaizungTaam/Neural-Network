#include <iostream>
#include <vector>
#include "../src/vector.hh"
#include "../src/matrix.hh"
#include "../src/utils.hh"

int main() {
  double a = 1.5, b = 1.5005;
  Vector u = {1.2, 2.4, 3.6, 4.8}, 
         v = {5.1, 3,8, 2,3, 9,9},
         w = {0.2, 0.5};
  Matrix m = {{1, 2, 3, 4},
              {5, 6, 7, 8}},
         n = {{1.2, 3.4, 5.6},
              {2.9, 3.7, 5.0},
              {7.7, 9.1, 1.1}},
         o = {{0.2, 0.5}, 
              {0.4, 0.7}};
  std::cout << nn::approx(a, b, 1e-3) << std::endl;
  std::cout << nn::exp(a) << std::endl;
  std::cout << nn::exp(u) << std::endl;
  std::cout << nn::exp(m) << std::endl;
  std::cout << nn::log(a) << std::endl;
  std::cout << nn::log(u) << std::endl;
  std::cout << nn::log(m) << std::endl;
  std::cout << nn::pow(a, b) << std::endl;
  std::cout << nn::pow(u, b) << std::endl;
  std::cout << nn::pow(m, b) << std::endl;
  std::cout << nn::sqrt(a) << std::endl;
  std::cout << nn::sqrt(u) << std::endl;
  std::cout << nn::sqrt(m) << std::endl;
  std::cout << nn::relu(a) << std::endl;
  std::cout << nn::relu(u) << std::endl;
  std::cout << nn::relu(m) << std::endl;
  std::cout << nn::logistic(a) << std::endl;
  std::cout << nn::logistic(u) << std::endl;
  std::cout << nn::logistic(m) << std::endl;
  std::cout << nn::tanh(a) << std::endl;
  std::cout << nn::tanh(u) << std::endl;
  std::cout << nn::tanh(m) << std::endl;
  std::cout << nn::softmax(u) << std::endl;
  std::cout << nn::softmax(m) << std::endl;
  std::cout << nn::d_logistic(m) << std::endl;
  std::cout << nn::activ_func(u, "logistic") << std::endl;
  std::cout << nn::activ_func(m, "logistic") << std::endl;
  std::cout << nn::d_activ_func(u, "logistic") << std::endl;
  std::cout << nn::d_activ_func(m, "logistic") << std::endl;
  std::cout << nn::convolve(u, w) << std::endl;
  std::cout << nn::convolve(n, o) << std::endl;
  return 0;
}