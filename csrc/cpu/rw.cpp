#include <torch/extension.h>

#include <iostream>

torch::Tensor walk(torch::Tensor edges, torch::Tensor target_nodes, float p, float q, int walk_length) {
  auto s = torch::sigmoid(edges);
  return (1 - s) * s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("walk", &walk, "walk");
}