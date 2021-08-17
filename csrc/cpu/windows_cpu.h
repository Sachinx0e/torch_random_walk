#include <torch/extension.h>

std::vector<torch::Tensor> to_windows_cpu(const torch::Tensor *walks,
                        const int window_size,
                        const int64_t num_nodes,
                        const int seed
                        );