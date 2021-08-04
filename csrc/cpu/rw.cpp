#include <torch/extension.h>

#include <iostream>

int64_t sample_neighbor(int64_t target_node, const torch::Tensor *row_ptr, const torch::Tensor *column_idx) {
  
  // get the row indices
  auto row_start = target_node;
  auto row_end = row_start + 1;

  // get the column indices
  auto column_start = (*row_ptr)[row_start].item().to<int64_t>();
  auto column_end = (*row_ptr)[row_end].item().to<int64_t>();

  // get the neighbors
  auto neighors = (*column_idx).slice(0,column_start,column_end,1);

  if(neighors.size(0) > 0){
    int64_t neighbor_idx = rand() % neighors.size(0);
    int64_t neighbor = neighors[neighbor_idx].item().to<int64_t>();
    return neighbor;
  }else{
    return -1;
  }

}

void uniform_walk(const torch::Tensor *walks, const torch::Tensor *row_ptr, const torch::Tensor *column_idx,const torch::Tensor *target_nodes, const int seed) {

    //set the seed
    srand(seed);

    // get the walk length
    int64_t walk_length = (*walks).size(1);
    
    // get the number of nodes
    int64_t num_nodes = (*target_nodes).size(0);

    // get the step size
    int grain_size = torch::internal::GRAIN_SIZE;

    // loop in parallel
    torch::parallel_for(0,num_nodes,grain_size,[&](int64_t node_start,int64_t node_end){
        for (int64_t node_index = node_start; node_index < node_end;node_index++) {
          
          // get the walk array for this node
          auto walks_for_node = (*walks)[node_index];
          
          // get the target node
          int64_t target_node = (*target_nodes)[node_index].item().to<int>();

          // add target node as the first node in walk
          walks_for_node[0] = target_node;

          // start walk
          int64_t previous_node = target_node;
          for (int64_t walk_step=1;walk_step < walk_length;walk_step++){
            // sample a neighor
            int64_t next_node = sample_neighbor(previous_node,row_ptr,column_idx);
            walks_for_node[walk_step] = next_node;

            // update previous node
            previous_node = next_node;

          }
      }
    });
}

torch::Tensor walk(const torch::Tensor *row_ptr, const torch::Tensor *column_idx, const torch::Tensor *target_nodes, const double p, const double q, const int walk_length, const int seed) {
  // construct a tensor to hold the walks
  auto walk_size = walk_length + 1;  
  auto walks = torch::empty({(*target_nodes).size(0),walk_size},torch::kInt64); 
  
  // perform walks
  if(p == 1.0 && q == 1.0){
    uniform_walk(&walks,row_ptr,column_idx,target_nodes,seed);
  }else{
    auto tmp = torch::ones({(*target_nodes).size(0),walk_length},torch::kInt64);
  }
  return walks;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("walk", &walk, "walk");
}