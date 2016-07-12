#ifndef INPUTLAYER_HH
#define INPUTLAYER_HH

#include <iostream>
#include <vector>
#include "matrix.hh"
#include "layer.hh"

class InputLayer : public Layer {
public:
  InputLayer() = default;
  InputLayer(const InputLayer &) = default;
  InputLayer(InputLayer &&) = default;
  InputLayer & operator=(const InputLayer &) = default;
  InputLayer & operator=(InputLayer &&) = default;
  ~InputLayer() = default;
  InputLayer(int);
  InputLayer(const Matrix &);
  std::vector<std::size_t> shape() const;
  void preprocess();
  friend std::istream & operator>>(std::istream & in, InputLayer &);
};

#endif  // input_layer.hh
