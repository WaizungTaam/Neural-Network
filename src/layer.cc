#include <vector>
#include <string>
#include "matrix.hh"
#include "layer.hh"

void Layer::input(const Matrix & mat_in) {
  // data_in.clear();
  data_in = mat_in;
}
Matrix Layer::output() {
  return data_in;
}
Matrix Layer::output(const Matrix & mat_in) {
  return mat_in;
}
Matrix Layer::output(const Matrix & mat_in, std::string) {
  return mat_in;
}
Matrix Layer::forward() {
  return data_in; 
}
Matrix Layer::forward(int idx_row) {
  return data_in(idx_row);
}
Matrix Layer::forward(int idx_r_begin, int idx_r_end) {
  return data_in(idx_r_begin, idx_r_end);
}
Matrix Layer::forward(const Matrix & mat_forward) {
  return mat_forward;
}
Matrix Layer::forward(const Matrix & mat_forward, std::string) {
  return mat_forward;
}
Matrix Layer::backward() {
  return data_in;
}
Matrix Layer::backward(int idx_row) {
  return data_in(idx_row);
}
Matrix Layer::backward(int idx_r_begin, int idx_r_end) {
  return data_in(idx_r_begin, idx_r_end);
}
Matrix Layer::backward(const Matrix & mat_backward) {
  return mat_backward;
}
void Layer::update(double learning_rate) {
}
void Layer::update(double learning_rate, double momentum) {
}
void Layer::supervise(const Matrix &) {
}
std::vector<std::size_t> Layer::shape() const {
  return data_in.shape();
}