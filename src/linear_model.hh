#ifndef LINEAR_MODEL_HH
#define LINEAR_MODEL_HH

#include <vector>
#include "layer.hh"
#include "matrix.hh"
#include "utils.hh"

class LinearModel {
public:
  LinearModel() = default;
  LinearModel(const LinearModel &) = default;
  LinearModel(LinearModel &&) = default;
  LinearModel & operator=(const LinearModel &) = default;
  LinearModel & operator=(LinearModel &&) = default;
  ~LinearModel() = default;
  LinearModel & push(Layer &);
  LinearModel & insert(Layer &, int);
  LinearModel & remove(int);
  LinearModel & replace(Layer &, int);
  void compile(const nn::param_list &);
  void input(const Matrix &);
  Matrix output();
  void supervise(const Matrix &);
  void train();
  std::vector<std::size_t> shape() const;
private:
  std::vector<Layer*> layers;
  nn::param_list parameters;
  std::size_t num_samples;
};

#endif  // linear_model.hh