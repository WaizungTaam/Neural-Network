#ifndef OUTPUT_LAYER_HH
#define OUTPUT_LAYER_HH

#include <vector>
#include <string>
#include "matrix.hh"
#include "layer.hh"

class OutputLayer : public Layer {
public:
  OutputLayer() = default;
  OutputLayer(const OutputLayer &) = default;
  OutputLayer(OutputLayer &&) = default;
  OutputLayer & operator=(const OutputLayer &) = default;
  OutputLayer & operator=(OutputLayer &&) = default;
  ~OutputLayer() = default;
  OutputLayer(int, int, std::string func="logistic");
  OutputLayer(int, int, const Matrix &, std::string func="logistic");
  OutputLayer(const Matrix &, const Matrix &, std::string func="logistic");
  OutputLayer(const Matrix &, const Matrix &, const Matrix &, std::string func="logistic");
  Matrix output();
  Matrix output(const Matrix &, std::string func="binary_step");
  Matrix forward();  
  Matrix forward(int);  
  Matrix forward(int, int);  
  Matrix forward(const Matrix &, std::string func="logistic");
  Matrix backward();  
  Matrix backward(int);
  Matrix backward(int, int);
  Matrix backward(const Matrix &);
  void update(double);
  void update(double, double);
  std::vector<std::size_t> shape() const;
  void supervise(const Matrix &);
private:
  Matrix weight;
  Matrix w_bias;
  Matrix delta_weight;
  Matrix delta_w_bias;
  Matrix output_given;
  // Matrix x_forward;
  Matrix local_field;
  Matrix local_gradient;
  std::string activ_func_name;
};

#endif  // output_layer.hh