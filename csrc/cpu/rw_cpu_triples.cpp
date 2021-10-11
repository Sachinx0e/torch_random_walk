#include "rw_cpu_triples.h"
#include <iostream>
#include <thread>
#include <random>
#include "../cuda/utils.cuh"
#include "cpu_utils.h"
#include "../utils.h"

namespace triples {

  RelationTail sample_neighbor(int64_t target_node,
                        const torch::TensorAccessor<int64_t,2> relation_tail_index,
                        const torch::TensorAccessor<int64_t,2> triples_indexed,
                        int64_t padding_index
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
        auto nbr_edge_index = sample_int(start_index,end_index);

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

  void uniform_walk_triples(const torch::Tensor *walks,
                    const torch::Tensor *triples_indexed,
                    const torch::Tensor *relation_tail_index,
                    const torch::Tensor *target_nodes,
                    const int64_t padding_idx,
                    const bool restart,
                    const int seed
                  ) {
      // seed
      srand(seed);

      // get the walk length
      int64_t walk_length = (*walks).size(1);
      
      // get the number of nodes
      int64_t num_nodes = (*target_nodes).size(0);


      // get the step size
      int grain_size = torch::internal::GRAIN_SIZE;

      // create accessors
      auto walks_accessor = walks->accessor<int64_t,2>();
      auto target_nodes_accessor = target_nodes->accessor<int64_t,1>();
      auto triples_indexed_accessor = triples_indexed->accessor<int64_t,2>();
      auto relation_tail_index_accessor = relation_tail_index->accessor<int64_t,2>();

      // loop in parallel
      torch::parallel_for(0,num_nodes,grain_size,[&](int64_t node_start,int64_t node_end){

          for (int64_t node_index = node_start; node_index < node_end;node_index++) {
            
            // get the walk array for this node
            auto walks_for_node = walks_accessor[node_index];
            
            // get the target node
            int64_t target_node = target_nodes_accessor[node_index];
            
            // add target node as the first node in walk
            walks_for_node[0] = target_node;

            // start walk
            int64_t previous_node = target_node;
            for (int64_t walk_step=1;walk_step < walk_length;walk_step=walk_step+2){
              // sample a neighor
              auto next_rt = sample_neighbor(previous_node,relation_tail_index_accessor,triples_indexed_accessor,padding_idx);
              walks_for_node[walk_step] = next_rt.relation;
              walks_for_node[walk_step+1] = next_rt.tail;
              
              // update previous node
              previous_node = next_rt.tail;

            }
        }
      });
  }

  torch::Tensor walk_triples_cpu(const torch::Tensor *triples_indexed,
                    const torch::Tensor *relation_tail_index,
                    const torch::Tensor *target_nodes,
                    const int walk_length,
                    const int64_t padding_idx,
                    const bool restart,
                    const int seed                  
                  ) {

    CHECK_CPU((*triples_indexed));
    CHECK_CPU((*relation_tail_index));
    CHECK_CPU((*target_nodes));

    // construct a tensor to hold the walks
    auto walk_size = (walk_length * 2) + 1;  
    auto walks = torch::empty({(*target_nodes).size(0),walk_size},torch::kInt64); 
    
    // perform walks
    uniform_walk_triples(&walks,triples_indexed,relation_tail_index,target_nodes,padding_idx,restart,seed);
    
    return walks;

  }
}

