#pragma once
#include <torch/extension.h>

namespace triples {
  torch::Tensor walk_triples_gpu(const torch::Tensor *triples_indexed,
                  const torch::Tensor *relation_tail_index,
                  const torch::Tensor *target_nodes,
                  const int walk_length,
                  const int64_t padding_idx,
                  const bool restart,
                  const int seed
                );
}