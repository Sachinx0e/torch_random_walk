#include "rw_cpu_edge_list.h"
#include <iostream>
#include <thread>
#include <random>
#include "../cuda/utils.cuh"
#include "cpu_utils.h"

int64_t sample_neighbor(int64_t target_node,
                        int64_t jump_node,
                        const torch::TensorAccessor<int64_t,2> node_edge_index,
                        const torch::TensorAccessor<int64_t,2> edge_list_indexed,
                        int64_t padding_index
                      ) {
  
  if(target_node != padding_index){
    // get the edge range for the target node
    auto start_index = node_edge_index[target_node][0];
    auto end_index = node_edge_index[target_node][1];

    // randomly select an index in this range
    if(start_index == -1 || end_index == -1){
      return padding_index;
    }else{
      auto nbr_edge_index = sample_int(start_index,end_index);

      // get the edge at this index
      auto nbr_id = edge_list_indexed[nbr_edge_index][1];

      return nbr_id;
    }
  }else{
    // restart the walk from jump
    return jump_node;
  }
}

bool is_neighbor(int64_t new_node,
                 int64_t previous_node,
                 const torch::TensorAccessor<int64_t,2> node_edge_index,
                 const torch::TensorAccessor<int64_t,2> edge_list_indexed) {
  
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

void uniform_walk_edge_list(const torch::Tensor *walks,
                  const torch::Tensor *edge_list_indexed,
                  const torch::Tensor *node_edge_index,
                  const torch::Tensor *target_nodes,
                  const int seed,
                  const int64_t padding_idx,
                  const bool restart
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

          // set the jump node according to restart policy
          int64_t jump_node = 0;
          if(restart == true){
            jump_node = target_node;
          }else{
            jump_node = padding_idx;
          }

          // add target node as the first node in walk
          walks_for_node[0] = target_node;

          // start walk
          int64_t previous_node = target_node;
          for (int64_t walk_step=1;walk_step < walk_length;walk_step++){
            // sample a neighor
            int64_t next_node = sample_neighbor(previous_node,jump_node,node_edge_index_accessor,edge_list_accessor,padding_idx);
            walks_for_node[walk_step] = next_node;

            // update previous node
            previous_node = next_node;

          }
      }
    });
}

void biased_walk_edge_list(const torch::Tensor *walks,
                  const torch::Tensor *edge_list_indexed,
                  const torch::Tensor *node_edge_index,
                  const torch::Tensor *target_nodes,
                  const double p,
                  const double q,
                  const int seed,
                  const int64_t padding_idx,
                  const bool restart
                  ) {
    
    // seed
    srand(seed);

    // get the walk length
    int64_t walk_length = (*walks).size(1);
    
    // get the number of nodes
    int64_t num_nodes = (*target_nodes).size(0);

    // get the step size
    int grain_size = torch::internal::GRAIN_SIZE / walk_length;

    // create accessors
    auto walks_accessor = walks->accessor<int64_t,2>();
    auto target_nodes_accessor = target_nodes->accessor<int64_t,1>();
    auto edge_list_accessor = edge_list_indexed->accessor<int64_t,2>();
    auto node_edge_index_accessor = node_edge_index->accessor<int64_t,2>();

    // normalize rejection probs
    double max_prob_init = fmax(1.0/p,1);
    double max_prob = fmax(max_prob_init,1.0/q);
    double prob_0 = 1.0/p/max_prob;
    double prob_1 = 1.0/max_prob;
    double prob_2 = 1.0/q/max_prob;

    
  
    // loop in parallel
    torch::parallel_for(0,num_nodes,grain_size,[&](int64_t node_start,int64_t node_end){
        for (int64_t node_index = node_start; node_index < node_end;node_index++) {
          
          // get the walk array for this node
          auto walks_for_node = walks_accessor[node_index];
          
          // get the target node
          int64_t target_node = target_nodes_accessor[node_index];

          // set the jump node according to restart policy
          int64_t jump_node = 0;
          if(restart == true){
            jump_node = target_node;
          }else{
            jump_node = padding_idx;
          }

          // add target node as the first node in walk
          walks_for_node[0] = target_node;

          // sample the first neighbor
          walks_for_node[1] = sample_neighbor(target_node,jump_node,node_edge_index_accessor,edge_list_accessor,padding_idx);

          // start walk
          int64_t previous_node = walks_for_node[1];
          for (int64_t walk_step=2;walk_step < walk_length;walk_step++){
            
            int64_t selected_node = -1;
            while(true) {
              // sample a new neighbor
              int64_t new_node = sample_neighbor(previous_node,jump_node,node_edge_index_accessor,edge_list_accessor,padding_idx);
              auto random_prob = ((double)rand()/(double)RAND_MAX);

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
              if(new_node == padding_idx){
                if(random_prob < prob_0){
                    selected_node = jump_node;
                    break;
                }
              }

              // else if new_node and t_node are neighbors i.e distance is 1
              else if(is_neighbor(new_node,t_node,node_edge_index_accessor,edge_list_accessor)){
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
    });
}

torch::Tensor walk_edge_list_cpu(const torch::Tensor *edge_list_indexed,
                  const torch::Tensor *node_edges_idx,
                  const torch::Tensor *target_nodes,
                  const double p,
                  const double q,
                  const int walk_length,
                  const int seed,
                  const int64_t padding_idx,
                  const bool restart                  
                ) {

  CHECK_CPU((*edge_list_indexed));
  CHECK_CPU((*node_edges_idx));
  CHECK_CPU((*target_nodes));

  // construct a tensor to hold the walks
  auto walk_size = walk_length + 1;  
  auto walks = torch::empty({(*target_nodes).size(0),walk_size},torch::kInt64); 
  
  // perform walks
  if(p == 1.0 && q == 1.0){
    uniform_walk_edge_list(&walks,edge_list_indexed,node_edges_idx,target_nodes,seed,padding_idx,restart);
  }else{
    biased_walk_edge_list(&walks,edge_list_indexed,node_edges_idx,target_nodes,p,q,seed,padding_idx,restart);
  }
  return walks;
}