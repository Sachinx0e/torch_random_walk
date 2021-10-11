#include "rw_cuda_triples.h"
#include <iostream>
#include <thread>
#include <ATen/cuda/CUDAContext.h>
#include <curand.h>
#include <curand_kernel.h>
#include "../utils.h"
#include "utils.cuh"

namespace triples {
  
  
  void uniform_walk_triples_gpu(const torch::Tensor *walks,
                    const torch::Tensor *triples_indexed,
                    const torch::Tensor *relation_tail_index,
                    const torch::Tensor *target_nodes,
                    const int64_t padding_idx,
                    const bool restart,
                    const int seed
                  ) {


  }

  torch::Tensor walk_triples_gpu(const torch::Tensor *triples_indexed,
                    const torch::Tensor *relation_tail_index,
                    const torch::Tensor *target_nodes,
                    const int walk_length,
                    const int64_t padding_idx,
                    const bool restart,
                    const int seed                  
                  ) {

    CHECK_CUDA((*triples_indexed));
    CHECK_CUDA((*relation_tail_index));
    CHECK_CUDA((*target_nodes));

    // construct a tensor to hold the walks
    auto walk_size = (walk_length * 2) + 1;  
    auto walks = torch::empty({(*target_nodes).size(0),walk_size},torch::kInt64); 
   
    
    
    return walks;

  }
}

