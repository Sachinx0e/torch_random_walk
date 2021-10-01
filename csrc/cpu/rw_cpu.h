#include <torch/extension.h>

torch::Tensor walk_cpu(const torch::Tensor *row_ptr,
                  const torch::Tensor *column_idx,
                  const torch::Tensor *target_nodes,
                  const double p,
                  const double q,
                  const int walk_length,
                  const int seed);

torch::Tensor walk_edge_list_cpu(const torch::Tensor *edge_list,
                  const torch::Tensor *node_range_mapping,
                  const torch::Tensor *target_nodes,
                  const double p,
                  const double q,
                  const int walk_length,
                  const int seed);