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

    auto nbr_idx = column_start + (curand(rand_state) % ((column_end) - column_start));

    // check bounds
    if(nbr_idx >= 0 && nbr_idx < col_length){
      auto neighbor = col_accessor[nbr_idx];
      return neighbor;
    }else{
      return target_node;
    } 
}

__device__ bool is_neighbor_gpu(int64_t new_node,
                 int64_t previous_node,
                 const torch::PackedTensorAccessor64<int64_t,1> row_accessor,
                 const torch::PackedTensorAccessor64<int64_t,1> col_accessor,
                 const int col_length) {
  
  // get the row indices
  auto row_start = previous_node;
  auto row_end = row_start + 1;

  // get the column indices
  auto column_start = row_accessor[row_start];
  auto column_end = row_accessor[row_end];

  // get the neighbors
  for(int64_t i = column_start;i<column_end;i++){
    auto node = col_accessor[i];
    if(node==new_node){
      return true;
    }
  }

  return false;

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

__global__ void biased_walk_gpu(const torch::PackedTensorAccessor64<int64_t,2> walks,
                            const torch::PackedTensorAccessor64<int64_t,1> row_ptr,
                            const torch::PackedTensorAccessor64<int64_t,1> col_idx,
                            const torch::PackedTensorAccessor64<int64_t,1> target_nodes,
                            const int walk_length,
                            const int num_nodes,
                            const int col_length,
                            const double p,
                            const double q,
                            const int seed) {
    
  // get the thread
  const auto thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  // seed rng
  curandState_t rand_state;
  curand_init(seed,thread_index,0,&rand_state);

  // normalize rejection probs
  double max_prob_init = fmax(1.0/p,1);
  double max_prob = fmax(max_prob_init,1.0/q);
  double prob_0 = 1.0/p/max_prob;
  double prob_1 = 1.0/max_prob;
  double prob_2 = 1.0/q/max_prob;


  // bound check
  if(thread_index < num_nodes) {
    // get the walk array for this node
    auto walks_for_node = walks[thread_index];
    
    // get the target node
    int64_t target_node = target_nodes[thread_index];

    // add target node as the first node in walk
    walks_for_node[0] = target_node;

    // sample the first neighbor
    walks_for_node[1] = sample_neighbor_gpu(target_node,row_ptr,col_idx,col_length,&rand_state);

    // start walk
    int64_t previous_node = walks_for_node[1];
    for(int64_t walk_step = 2; walk_step< walk_length;walk_step++){
      int64_t selected_node = -1;
      
      // rejection sampling
      while(true) {

        // sample a new neighbor
        int64_t new_node = sample_neighbor_gpu(previous_node,row_ptr,col_idx,col_length,&rand_state);
        auto random_prob = curand_uniform(&rand_state);

        // t_node
        int64_t t_node = walks_for_node[walk_step-2];
      
        // new_node is the same as previous to previous node, so go back.
        if(new_node == t_node) {
          if(random_prob < prob_0){
              selected_node = new_node;
              break;
          }
        }

        // else if new_node and t_node are neighbors i.e distance is 1
        else if(is_neighbor_gpu(new_node,t_node,row_ptr,col_idx,col_length)){
          if(random_prob < prob_1) {
            selected_node = new_node;
            break;
          }
        }

        // else distance is 2
        else if(random_prob < prob_2){
          selected_node = new_node;
          break;
        }

        

      } // end while
      walks_for_node[walk_step] = selected_node;           
      previous_node = selected_node;
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
  
  auto stream = at::cuda::getCurrentCUDAStream();

  // perform walks
  if(p == 1.0 && q == 1.0){
    uniform_walk_gpu<<<NUM_BLOCKS,NUM_THREADS,0,stream>>>(walks_accessor,
                                                row_ptr_accessor,
                                                col_idx_accessor,
                                                target_nodes_accesor,
                                                walk_size,
                                                num_nodes,
                                                col_length,
                                                seed);
  }else{
    biased_walk_gpu<<<NUM_BLOCKS,NUM_THREADS,0,stream>>>(walks_accessor,
                                                row_ptr_accessor,
                                                col_idx_accessor,
                                                target_nodes_accesor,
                                                walk_size,
                                                num_nodes,
                                                col_length,
                                                p,
                                                q,
                                                seed);
  }
  return walks;
}