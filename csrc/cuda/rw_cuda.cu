#include "rw_cuda.h"
#include "utils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <curand.h>
#include <curand_kernel.h>

__device__ int64_t sample_neighbor_gpu(int64_t target_node,
                        const torch::PackedTensorAccessor64<int64_t,1> row_accessor,
                        const torch::PackedTensorAccessor64<int64_t,1> col_accessor,
                        int64_t col_length,
                        curandState_t* rand_state
                        ) {
    // get the row indices
    auto row_start = target_node;
    auto row_end = row_start + 1;

    // get the column indices
    auto column_start = row_accessor[row_start];
    auto column_end = row_accessor[row_end];

    auto nbr_idx = column_start + ( curand(rand_state) % ((column_end) - column_start));

    // check bounds
    if(nbr_idx >= 0 && nbr_idx < col_length){
      auto neighbor = col_accessor[nbr_idx];
      return neighbor;
    }else{
      return -1;
    } 
}

__global__ void uniform_walk_gpu(const torch::PackedTensorAccessor64<int64_t,2> walks,
                            const torch::PackedTensorAccessor64<int64_t,1> row_ptr,
                            const torch::PackedTensorAccessor64<int64_t,1> col_idx,
                            const torch::PackedTensorAccessor64<int64_t,1> target_nodes,
                            const int walk_length,
                            const int num_nodes,
                            const int col_length,
                            const int seed) {
    
    // get the thread
    const auto thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    // seed rng
    curandState_t rand_state;
    curand_init(seed,thread_index,0,&rand_state);
        
    // bound check
    if(thread_index < num_nodes) {
        // get the walk array for this node
        auto walks_for_node = walks[thread_index];
              
        // get the target node
        int64_t target_node = target_nodes[thread_index];

        // add target node as the first node in walk
        walks_for_node[0] = target_node;

        // start walk
        int64_t previous_node = target_node;
        for (int64_t walk_step=1;walk_step < walk_length;walk_step++){
          // sample a neighor
          int64_t next_node = sample_neighbor_gpu(previous_node,row_ptr,col_idx,col_length,&rand_state);
          walks_for_node[walk_step] = next_node;

          // update previous node
          previous_node = next_node;

        }
    }   
}

torch::Tensor walk_gpu(const torch::Tensor *row_ptr,
                  const torch::Tensor *column_idx,
                  const torch::Tensor *target_nodes,
                  const double p,
                  const double q,
                  const int walk_length,
                  const int seed) {

  CHECK_CUDA((*row_ptr));
  CHECK_CUDA((*column_idx));
  CHECK_CUDA((*target_nodes));

  cudaSetDevice(row_ptr->device().index());

  // construct a tensor to hold the walks
  auto walk_size = walk_length + 1;
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA,row_ptr->device().index());  
  auto walks = torch::empty({(*target_nodes).size(0),walk_size},options); 
  
  // create accessors
  auto walks_accessor = walks.packed_accessor64<int64_t,2>();
  auto row_ptr_accessor = row_ptr->packed_accessor64<int64_t,1>();
  auto col_idx_accessor = column_idx->packed_accessor64<int64_t,1>();
  auto target_nodes_accesor = target_nodes->packed_accessor64<int64_t,1>();

  // get the number of nodes
  int64_t num_nodes = (*target_nodes).size(0);

  // get col length
  int64_t col_length = column_idx->size(0);

  // Thread block size
  int NUM_THREADS = 1024;

  // Grid size
  int NUM_BLOCKS = int((num_nodes + NUM_THREADS - 1)/NUM_THREADS);
  
  // perform walks
  if(p == 1.0 && q == 1.0){
    uniform_walk_gpu<<<NUM_BLOCKS,NUM_THREADS>>>(walks_accessor,
                                                row_ptr_accessor,
                                                col_idx_accessor,
                                                target_nodes_accesor,
                                                walk_size,
                                                num_nodes,
                                                col_length,
                                                seed);
  }else{
    //biased_walk(&walks,row_ptr,column_idx,target_nodes,p,q,seed);
  }
  return walks;
}