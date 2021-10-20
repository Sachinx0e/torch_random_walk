#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor, at::Tensor> to_windows_cpu(const torch::Tensor *walks,
                        const int window_size,
                        const int64_t num_nodes,
                        const int seed
                        );

std::tuple<at::Tensor, at::Tensor, at::Tensor> to_windows_cbow_cpu(const torch::Tensor *walks,
                        const int window_size,
                        const int64_t num_nodes,
                        const int seed
                        );

std::tuple<at::Tensor, at::Tensor, at::Tensor> to_windows_triples_cpu(const torch::Tensor *walks,
                        const int window_size,
                        const int64_t num_nodes,
                        const int64_t padding_idx,
                        const torch::Tensor *triples,
                        const int seed
                        );

std::tuple<at::Tensor, at::Tensor, at::Tensor> to_windows_triples_cbow_cpu(const torch::Tensor *walks,
                        const int window_size,
                        const int64_t num_nodes,
                        const int64_t padding_idx,
                        const torch::Tensor *triples,
                        const int seed
                    );