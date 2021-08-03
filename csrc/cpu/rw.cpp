#include <torch/extension.h>

#include <iostream>

void uniform_walk(const torch::Tensor *walks, const torch::Tensor *edges,const torch::Tensor *target_nodes,const int walk_length) {
    // get the number of nodes
    int num_nodes = (*target_nodes).size(0);

    // get the step size
    int grain_size = at::internal::GRAIN_SIZE / walk_length;

    // loop in parallel
    at::parallel_for(0,num_nodes,grain_size,[&](int node_start,int node_end){
        for (int node_index = node_start; node_index < node_end;node_index++) {
          auto walks_for_node = (*walks)[node_index];
          for (int walk_step=0;walk_step < walk_length;walk_step++){
            walks_for_node[walk_step] = 1;
          }
      }
    });
}

torch::Tensor walk(const torch::Tensor *edges, const torch::Tensor *target_nodes, const double p, const double q, const int walk_length) {
  // construct a tensor to hold the walks  
  auto walks = at::empty({(*target_nodes).size(0),walk_length},at::kInt); 
  
  // perform walks
  if(p == 1.0 && q == 1.0){
    uniform_walk(&walks,edges,target_nodes,walk_length);
  }else{
    auto tmp = at::ones({(*target_nodes).size(0),walk_length},at::kInt);
  }
  return walks;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("walk", &walk, "walk");
}