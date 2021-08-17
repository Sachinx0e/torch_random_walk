#include <torch/extension.h>

std::vector<torch::Tensor> to_windows_cpu(const torch::Tensor *row_ptr,
                        const int window_size,
                        const int num_nodes);