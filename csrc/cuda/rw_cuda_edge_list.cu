#include "rw_cuda_edge_list.h"
#include "utils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <curand.h>
#include <curand_kernel.h>


__device__ int64_t sample_int(int64_t start, int64_t end,curandState_t* rand_state){
    auto sampled_int = start + (curand(rand_state) % (((end + 1) - start)));
    return sampled_int;
}

__device__ int64_t sample_neighbor_gpu(int64_t target_node,
                        int64_t jump_node,
                        int64_t padding_index,
                        const torch::PackedTensorAccessor64<int64_t,2> node_edge_index,
                        const torch::PackedTensorAccessor64<int64_t,2> edge_list_indexed,
                        curandState_t* rand_state
                        ) {
    if(target_node != padding_index){
    // get the edge range for the target node
    auto start_index = node_edge_index[target_node][0];
    auto end_index = node_edge_index[target_node][1];

    // randomly select an index in this range
    if(start_index == -1 || end_index == -1){
      return padding_index;
    }else{
      auto nbr_edge_index = sample_int(start_index,end_index,rand_state);

      // get the edge at this index
      auto nbr_id = edge_list_indexed[nbr_edge_index][1];

      return nbr_id;
    }
  }else{
    // restart the walk from first node
    return jump_node;
  } 
}

__global__ void uniform_walk_edge_list_gpu(const torch::PackedTensorAccessor64<int64_t,2> walks,
                  const torch::PackedTensorAccessor64<int64_t,2> edge_list_indexed_accessor,
                  const torch::PackedTensorAccessor64<int64_t,2> node_edges_index_accessor,
                  const torch::PackedTensorAccessor64<int64_t,1> target_nodes_accesor,
                  const int walk_length,
                  const int64_t padding_index,
                  const int64_t num_nodes,
                  const int seed,
                  const bool restart
                  ) {
    
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
        int64_t target_node = target_nodes_accesor[thread_index];

        // set the jump node according to restart policy
        int64_t jump_node = 0;
        if(restart == true){
          jump_node = target_node;
        }else{
          jump_node = padding_index;
        }

        // add target node as the first node in walk
        walks_for_node[0] = target_node;

        // start walk
        int64_t previous_node = target_node;
        for (int64_t walk_step=1;walk_step < walk_length;walk_step++){
          // sample a neighor
          int64_t next_node = sample_neighbor_gpu(previous_node,
                                                  jump_node,
                                                  padding_index,
                                                  node_edges_index_accessor,
                                                  edge_list_indexed_accessor,
                                                  &rand_state);
          walks_for_node[walk_step] = next_node;

          // update previous node
          previous_node = next_node;

        }
    }
}

__device__ bool is_neighbor(int64_t new_node,
                 int64_t previous_node,
                 const torch::PackedTensorAccessor64<int64_t,2> node_edge_index,
                 const torch::PackedTensorAccessor64<int64_t,2> edge_list_indexed) {
  
  // get the edge range for the target node
  auto start_index = node_edge_index[previous_node][0];
  auto end_index = node_edge_index[previous_node][1];

  // randomly select an index in this range
  if(start_index == -1 || end_index == -1){
    return false;
  }else{

    for(int64_t i = start_index;i<end_index;i++){
      auto node = edge_list_indexed[i][1];
      if(node==new_node){
        return true;
      }
    }

    return false;

  }

}


__global__ void biased_walk_edge_list_gpu(const torch::PackedTensorAccessor64<int64_t,2> walks,
                  const torch::PackedTensorAccessor64<int64_t,2> edge_list_indexed_accessor,
                  const torch::PackedTensorAccessor64<int64_t,2> node_edges_index_accessor,
                  const torch::PackedTensorAccessor64<int64_t,1> target_nodes_accesor,
                  const double p,
                  const double q,
                  const int walk_length,
                  const int64_t padding_index,
                  const int64_t num_nodes,
                  const int seed,
                  const bool restart
                  ) {
    
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
    int64_t target_node = target_nodes_accesor[thread_index];

    // set the jump node according to restart policy
    int64_t jump_node = 0;
    if(restart == true){
      jump_node = target_node;
    }else{
      jump_node = padding_index;
    }

    // add target node as the first node in walk
    walks_for_node[0] = target_node;

    // sample the first neighbor
    walks_for_node[1] = sample_neighbor_gpu(target_node,
                                            jump_node,
                                            padding_index,
                                            node_edges_index_accessor,
                                            edge_list_indexed_accessor,
                                            &rand_state);

    // start walk
    int64_t previous_node = walks_for_node[1];
    for(int64_t walk_step = 2; walk_step< walk_length;walk_step++){
      int64_t selected_node = -1;
      
      // rejection sampling
      while(true) {

        // sample a new neighbor
        int64_t new_node = sample_neighbor_gpu(previous_node,
                                                  jump_node,
                                                  padding_index,
                                                  node_edges_index_accessor,
                                                  edge_list_indexed_accessor,
                                                  &rand_state);

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

        // if new_node is a padding_idx then restart
        if(new_node == padding_index){
            if(random_prob < prob_0){
                selected_node = jump_node;
                break;
            }
        }

        // else if new_node and t_node are neighbors i.e distance is 1
        else if(is_neighbor(new_node,t_node,node_edges_index_accessor,edge_list_indexed_accessor)){
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


torch::Tensor walk_edge_list_gpu(const torch::Tensor *edge_list_indexed,
                  const torch::Tensor *node_edges_idx,
                  const torch::Tensor *target_nodes,
                  const double p,
                  const double q,
                  const int walk_length,
                  const int seed,
                  const int64_t padding_idx,
                  const bool restart                  
                ) {

  CHECK_CUDA((*edge_list_indexed));
  CHECK_CUDA((*node_edges_idx));
  CHECK_CUDA((*target_nodes));
  
  // construct a tensor to hold the walks
  auto walk_size = walk_length + 1;
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA,node_edges_idx->device().index());  
  auto walks = torch::empty({(*target_nodes).size(0),walk_size},options); 
  
  // create accessors
  auto walks_accessor = walks.packed_accessor64<int64_t,2>();
  auto edge_list_indexed_accessor = edge_list_indexed->packed_accessor64<int64_t,2>();
  auto node_edges_index_accessor = node_edges_idx->packed_accessor64<int64_t,2>();
  auto target_nodes_accesor = target_nodes->packed_accessor64<int64_t,1>();

  // get the number of nodes
  int64_t num_nodes = (*target_nodes).size(0);

  // Thread block size
  int NUM_THREADS = 1024;

  // Grid size
  int NUM_BLOCKS = int((num_nodes + NUM_THREADS - 1)/NUM_THREADS);
  
  // active stream
  auto stream = at::cuda::getCurrentCUDAStream();
  
  // perform walks
  if(p == 1.0 && q == 1.0){
    uniform_walk_edge_list_gpu<<<NUM_BLOCKS,NUM_THREADS,0,stream>>>(walks_accessor,
                                                                    edge_list_indexed_accessor,
                                                                    node_edges_index_accessor,
                                                                    target_nodes_accesor,
                                                                    walk_size,
                                                                    padding_idx,
                                                                    num_nodes,
                                                                    seed,
                                                                    restart
                                                                );
  }else{
    biased_walk_edge_list_gpu<<<NUM_BLOCKS,NUM_THREADS,0,stream>>>(walks_accessor,
                                                                    edge_list_indexed_accessor,
                                                                    node_edges_index_accessor,
                                                                    target_nodes_accesor,
                                                                    p,
                                                                    q,
                                                                    walk_size,
                                                                    padding_idx,
                                                                    num_nodes,
                                                                    seed,
                                                                    restart
                                                                );
  }
  return walks;
}