#ifndef PERCEPTRON_HH
#define PERCEPTRON_HH

#include <iostream>
#include "matrix.hh"

class Perceptron {
public:
  Perceptron();
  Perceptron(int, int);
  Perceptron(const Matrix &);
  Perceptron(const Perceptron &);
  Perceptron & operator=(const Perceptron &);
  void train(const Matrix &, const Matrix &, int, int, double);
  Matrix predict(const Matrix &);
  friend std::ostream & operator<<(std::ostream &, const Perceptron &);
private:
  Matrix weight;
};

#endif  // perceptron.hh