#include <vector>
#include <iostream>
#include "linear_model.hh"
#include "layer.hh"
#include "input_layer.hh"
#include "hidden_layer.hh"
#include "output_layer.hh"
#include "matrix.hh"
#include "utils.hh"

LinearModel & LinearModel::push(Layer & layer_push) {
  layers.push_back(&layer_push);
  return *this;
}
LinearModel & LinearModel::insert(Layer & layer_insert, int index) {
  layers.insert(layers.begin() + index, &layer_insert);
  return *this;
}
LinearModel & LinearModel::remove(int index) {
  layers.erase(layers.begin() + index);
  return *this;
}
LinearModel & LinearModel::replace(Layer & layer_replace, int index) {
  remove(index);
  insert(layer_replace, index);
}
void LinearModel::compile(const nn::param_list & params_init) {
  parameters = params_init;
  int idx_layer;
  for (idx_layer = 0; idx_layer < layers.size() - 1; ++idx_layer) {
    if (layers[idx_layer] -> shape()[1] != layers[idx_layer + 1] -> shape()[0]) {
      throw "LinearModel compile error: Inconsistent shape.";
    }
  }
}
void LinearModel::input(const Matrix & mat_in) {
  layers[0] -> input(mat_in);
  num_samples = mat_in.shape()[0];
}
Matrix LinearModel::output() {
  Matrix mat_out = layers[0] -> forward();
  int idx_layer;
  for (idx_layer = 1; idx_layer < layers.size() - 1; ++idx_layer) {
    mat_out = layers[idx_layer] -> forward(mat_out, parameters.activ_func);
  }
  mat_out = layers[layers.size() - 1] -> output(mat_out, parameters.output_func);
  return mat_out;
}
void LinearModel::supervise(const Matrix & output_given) {
  layers[layers.size() - 1] -> supervise(output_given);
}
void LinearModel::train() {
  int num_batches = (int)(num_samples / parameters.batch_size),
      idx_epoch, idx_batch, idx_batch_begin, idx_batch_end, idx_layer;
  Matrix output_pred, gradient_back;
  if (num_batches * parameters.batch_size != num_samples) {
    ++num_batches;
  }
  for (idx_epoch = 0; idx_epoch < parameters.num_epochs; ++idx_epoch) {
    for (idx_batch = 0; idx_batch < num_batches; ++idx_batch) {
      idx_batch_begin = idx_batch * parameters.batch_size;
      idx_batch_end = (idx_batch + 1) * parameters.batch_size;
      if (idx_batch == num_batches - 1) {
        idx_batch_end = num_samples;
      }
      std::cout << idx_epoch << " " << idx_batch << " ";
      output_pred = layers[0] -> forward(idx_batch_begin, idx_batch_end);
      for (idx_layer = 1; idx_layer < layers.size(); ++idx_layer) {
        output_pred = layers[idx_layer] -> forward(output_pred, parameters.activ_func);
      }
      gradient_back = layers[layers.size() - 1] -> backward(
        idx_batch_begin, idx_batch_end);
      for (idx_layer = layers.size() - 2; idx_layer > 0; --idx_layer) {
        gradient_back = layers[idx_layer] -> backward(gradient_back);
      }
      for (idx_layer = 1; idx_layer < layers.size(); ++idx_layer) {
        layers[idx_layer] -> update(parameters.learning_rate, 
                                    parameters.momentum);
      }
    }
  }
}
std::vector<std::size_t> LinearModel::shape() const {
  std::vector<std::size_t> shape_vec;
  shape_vec.push_back(layers.size());
  return shape_vec;
}