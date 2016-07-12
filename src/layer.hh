#ifndef LAYER_HH
#define LAYER_HH

#include <vector>
#include <string>
#include "matrix.hh"

class Layer {
public:
  Layer() = default;
  Layer(const Layer &) = default;
  Layer(Layer &&) = default;
  Layer & operator=(const Layer &) = default;
  Layer & operator=(Layer &&) = default;
  virtual ~Layer() = default;
  virtual void input(const Matrix &);
  virtual Matrix output();
  virtual Matrix output(const Matrix &);
  virtual Matrix output(const Matrix &, std::string);
  virtual Matrix forward();
  virtual Matrix forward(int);
  virtual Matrix forward(int, int);
  virtual Matrix forward(const Matrix &);
  virtual Matrix forward(const Matrix &, std::string);
  virtual Matrix backward();
  virtual Matrix backward(int);
  virtual Matrix backward(int, int);
  virtual Matrix backward(const Matrix &);
  virtual void update(double);
  virtual void update(double, double);
  virtual void supervise(const Matrix &);
  std::vector<std::size_t> shape() const;
protected:
  Matrix data_in;
};

#endif  // layer.hh