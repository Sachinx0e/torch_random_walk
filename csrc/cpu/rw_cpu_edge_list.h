#pragma once

#include <torch/extension.h>
torch::Tensor walk_edge_list_cpu(const torch::Tensor *edge_list,
                  const torch::Tensor *node_range_mapping,
                  const torch::Tensor *target_nodes,
                  const double p,
                  const double q,
                  const int walk_length,
                  const int seed,
                  const int64_t padding_idx,
                  const bool restart
                );