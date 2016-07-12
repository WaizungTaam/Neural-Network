#include "matrix.hh"
#include "perceptron.hh"

Perceptron::Perceptron() {
}
Perceptron::Perceptron(int dim_in, int dim_out) {
  Matrix w_init(dim_in, dim_out);
  weight = w_init;
}
Perceptron::Perceptron(const Matrix & w_init) {
  weight = w_init;
}
Perceptron::Perceptron(const Perceptron & p) {
  weight = p.weight;
} 
Perceptron & Perceptron::operator=(const Perceptron & p) {
  weight = p.weight;
  return *this;
}
void Perceptron::train(const Matrix & X, const Matrix & Y,
                       int num_epochs, int batch_size, 
                       double learning_rate) {
  if (X.shape()[0] != Y.shape()[0]) {
    throw "Inconsistent shape";
  }
  if (weight.shape()[0] == 0) {
    Matrix w_init(X.shape()[1], Y.shape()[1]);
    weight = w_init;
  } else {
    if (X.shape()[1] != weight.shape()[0] || 
        Y.shape()[1] != weight.shape()[1]) {
      throw "Inconsistent shape";
    }
  }
  int num_samples = X.shape()[0],
      num_batches = (int)(num_samples / batch_size),
      idx_epoch, idx_batch;
  if (num_batches * batch_size != num_samples) {
    ++num_batches;
  }
  int idx_batch_begin, idx_batch_end;
  for (idx_epoch = 0; idx_epoch < num_epochs; ++idx_epoch) {
    for (idx_batch = 0; idx_batch < num_batches; ++idx_batch) {
      idx_batch_begin = idx_batch * batch_size;
      idx_batch_end = (idx_batch + 1) * batch_size;
      if (idx_batch == num_batches - 1) {
        idx_batch_end = num_samples;
      }
      Matrix x = X(idx_batch_begin, idx_batch_end);
      Matrix y = Y(idx_batch_begin, idx_batch_end);
      Matrix y_pred = predict(x);
      weight += learning_rate * x.T() * (y - y_pred);
    }
  }
}
Matrix Perceptron::predict(const Matrix & X) {
  if (weight.shape()[0] != X.shape()[1]) {
    throw "Inconsistent shape";
  }
  Matrix res = ((X * weight) >= 0);
  return res;
}
std::ostream & operator<<(std::ostream & out, const Perceptron & p) {
  out << p.weight;
  return out;
}
