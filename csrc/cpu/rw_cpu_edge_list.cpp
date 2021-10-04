#include "rw_cpu_edge_list.h"
#include <iostream>
#include <thread>
#include <random>
#include "../cuda/utils.cuh"
#include "cpu_utils.h"

int64_t sample_neighbor(int64_t target_node,
                        const torch::TensorAccessor<int64_t,2> node_edge_index,
                        const torch::TensorAccessor<int64_t,2> edge_list_indexed,
                        int64_t padding_index
                      ) {
  
  if(target_node != padding_index){
    // get the edge range for the target node
    auto start_index = node_edge_index[target_node][0];
    auto end_index = node_edge_index[target_node][1];

    // randomly select an index in this range
    auto nbr_edge_index = 0;
    if(start_index == -1 || end_index == -1){
      return padding_index;
    }else{
      auto nbr_edge_index = sample_int(start_index,end_index);

      // get the edge at this index
      auto nbr_id = edge_list_indexed[nbr_edge_index][1];

      return nbr_id;
    }
  }else{
    return padding_index;
  }

}

void uniform_walk_edge_list(const torch::Tensor *walks,
                  const torch::Tensor *edge_list_indexed,
                  const torch::Tensor *node_edge_index,
                  const torch::Tensor *target_nodes,
                  const int seed,
                  const int64_t padding_idx
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
    auto edge_list_accessor = edge_list_indexed->accessor<int64_t,2>();
    auto node_edge_index_accessor = node_edge_index->accessor<int64_t,2>();

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
          for (int64_t walk_step=1;walk_step < walk_length;walk_step++){
            // sample a neighor
            int64_t next_node = sample_neighbor(previous_node,node_edge_index_accessor,edge_list_accessor,padding_idx);
            walks_for_node[walk_step] = next_node;

            // update previous node
            previous_node = next_node;

          }
      }
    });
}

torch::Tensor walk_edge_list_cpu(const torch::Tensor *edge_list_indexed,
                  const torch::Tensor *node_edges_idx,
                  const torch::Tensor *target_nodes,
                  const double p,
                  const double q,
                  const int walk_length,
                  const int seed,
                  const int64_t padding_idx                  
                ) {

  CHECK_CPU((*edge_list_indexed));
  CHECK_CPU((*node_edges_idx));
  CHECK_CPU((*target_nodes));

  // construct a tensor to hold the walks
  auto walk_size = walk_length + 1;  
  auto walks = torch::empty({(*target_nodes).size(0),walk_size},torch::kInt64); 
  
  // perform walks
  if(p == 1.0 && q == 1.0){
    uniform_walk_edge_list(&walks,edge_list_indexed,node_edges_idx,target_nodes,seed,padding_idx);
  }else{
    //biased_walk(&walks,row_ptr,column_idx,target_nodes,p,q,seed);
  }
  return walks;
}