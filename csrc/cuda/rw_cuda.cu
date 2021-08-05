#include "rw_cuda.h"
#include "utils.cuh"
#include <ATen/cuda/CUDAContext.h>

__device__ int64_t sample_neighbor_gpu(int64_t target_node,
                        const torch::PackedTensorAccessor64<int64_t,1> row_accessor,
                        const torch::PackedTensorAccessor64<int64_t,1> col_accessor,
                        int64_t col_length
                        ) {
    return 1;  
}

__global__ void uniform_walk_gpu(const torch::PackedTensorAccessor64<int64_t,2> *walks,
                            const torch::PackedTensorAccessor64<int64_t,1> *row_ptr,
                            const torch::PackedTensorAccessor64<int64_t,1> *column_idx,
                            const torch::PackedTensorAccessor64<int64_t,1> *target_nodes,
                            const int seed) {
    
    // get the thread
    const auto thread_index = blockIdx.x * blockDim.x + threadIdx.x;

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

  // Thread block size
  int NUM_THREADS = 256;

  // Grid size
  int NUM_BLOCKS = int((num_nodes + NUM_THREADS - 1)/NUM_THREADS);
  
  std::cout << "Num blocks: " + std::to_string(NUM_BLOCKS) << std::endl;

  // perform walks
  if(p == 1.0 && q == 1.0){
    uniform_walk_gpu<<<NUM_BLOCKS,NUM_THREADS>>>(&walks_accessor,
                                                &row_ptr_accessor,
                                                &col_idx_accessor,
                                                &target_nodes_accesor,
                                                seed);
  }else{
    //biased_walk(&walks,row_ptr,column_idx,target_nodes,p,q,seed);
  }
  return walks;
}