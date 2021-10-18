#pragma once

#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor, at::Tensor> to_windows_gpu(const torch::Tensor *walks,
                        const int window_size,
                        const int64_t num_nodes,
                        const int seed
                        );

std::tuple<at::Tensor, at::Tensor, at::Tensor> to_windows_triples_gpu(const torch::Tensor *walks,
                        const int window_size,
                        const int64_t num_nodes,
                        const int64_t padding_idx,
                        const torch::Tensor *triples,
                        const int seed
                        );