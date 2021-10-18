#include "rw_cuda_triples.h"
#include <iostream>
#include <thread>
#include <ATen/cuda/CUDAContext.h>
#include <curand.h>
#include <curand_kernel.h>
#include "../utils.h"
#include "utils.cuh"


namespace triples {
  
    __device__ RelationTail sample_neighbor_gpu(int64_t target_node,
                        int64_t padding_index,
                        const torch::PackedTensorAccessor64<int64_t,2> relation_tail_index,
                        const torch::PackedTensorAccessor64<int64_t,2> triples_indexed,
                        curandState_t* rand_state
                        ) {
    
        RelationTail rt;
    
        // if target node is padding index, then jump node becomes target node
        if(target_node != padding_index){
    
        // get the edge range for the target node
        auto start_index = relation_tail_index[target_node][0];
        auto end_index = relation_tail_index[target_node][1];
        // randomly select an index in this range
        if(start_index == -1 || end_index == -1){
            rt.relation = padding_index;
            rt.tail = padding_index;
        }else{
            auto nbr_edge_index = sample_int_gpu(start_index,end_index,rand_state);
    
            // get the edge at this index
            rt.relation = triples_indexed[nbr_edge_index][1];
            rt.tail = triples_indexed[nbr_edge_index][2];
        }
    
        }else{
            rt.relation = padding_index;
            rt.tail = padding_index;
        }
        
        return rt;
    
    }
    
    __global__ void uniform_walk_triples_gpu(const torch::PackedTensorAccessor64<int64_t,2> walks,
                    const torch::PackedTensorAccessor64<int64_t,2> triples_indexed_accessor,
                    const torch::PackedTensorAccessor64<int64_t,2> relation_tail_index_accessor,
                    const torch::PackedTensorAccessor64<int64_t,1> target_nodes_accesor,
                    const int walk_length,
                    const int64_t padding_idx,
                    const int64_t num_nodes,
                    curandState* rand_states
                    ) {
    
        // get the thread
        const auto thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    
   
        // bound check
        if(thread_index < num_nodes) {
            // rng
            auto rand_state = rand_states[thread_index];
     
            // get the walk array for this node
            auto walks_for_node = walks[thread_index];
                    
            // get the target node
            int64_t target_node = target_nodes_accesor[thread_index];
    
            // add target node as the first node in walk
            walks_for_node[0] = target_node;
    
            // start walk
            int64_t previous_node = target_node;
            for (int64_t walk_step=1;walk_step < walk_length;walk_step=walk_step+2){
                // sample a neighor
                auto next_rt = sample_neighbor_gpu(previous_node,
                                                        padding_idx,
                                                        relation_tail_index_accessor,
                                                        triples_indexed_accessor,
                                                        &rand_state);

                walks_for_node[walk_step] = next_rt.relation;
                walks_for_node[walk_step+1] = next_rt.tail;
              
                // update previous node
                previous_node = next_rt.tail;
    
            }
        }
    
    }

    __global__ void init_rand_states(const int64_t seed, curandState *states) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(seed, idx, 0, &states[idx]);
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
        auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA,target_nodes->device().index());  
        auto walks = torch::empty({(*target_nodes).size(0),walk_size},options); 
        
        // create accessors
        auto walks_accessor = walks.packed_accessor64<int64_t,2>();
        auto triples_indexed_accessor = triples_indexed->packed_accessor64<int64_t,2>();
        auto relation_tail_index_accessor = relation_tail_index->packed_accessor64<int64_t,2>();
        auto target_nodes_accesor = target_nodes->packed_accessor64<int64_t,1>();

        // get the number of nodes
        int64_t num_nodes = (*target_nodes).size(0);

        // Thread block size
        int NUM_THREADS = 1024;

        // Grid size
        int NUM_BLOCKS = int((num_nodes + NUM_THREADS - 1)/NUM_THREADS);

        // active stream
        auto stream = at::cuda::getCurrentCUDAStream();

        // random states
        curandState *rand_states;
        cudaMalloc(&rand_states, NUM_THREADS * NUM_BLOCKS);

        int64_t actual_seed = 0;
        if(seed==0){
            actual_seed = time(NULL);
        }else{
            actual_seed = actual_seed;
        }

        // init states
        init_rand_states<<<NUM_BLOCKS, NUM_THREADS,0,stream>>>(actual_seed,rand_states);
                                
        // perform walks
        uniform_walk_triples_gpu<<<NUM_BLOCKS,NUM_THREADS,0,stream>>>(walks_accessor,
                                                                        triples_indexed_accessor,
                                                                        relation_tail_index_accessor,
                                                                        target_nodes_accesor,
                                                                        walk_size,
                                                                        padding_idx,
                                                                        num_nodes,
                                                                        rand_states
                                                                    );

        // delete rand states
        cudaFree(rand_states);
        
        return walks;
    
    }
}



