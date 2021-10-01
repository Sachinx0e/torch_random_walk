#include <torch/extension.h>
#include "cpu/rw_cpu.h"
#include "cuda/rw_cuda.h"
#include "cpu/windows_cpu.h"
#include "cuda/windows_cuda.h"

torch::Tensor walk(const torch::Tensor *row_ptr,
                  const torch::Tensor *column_idx,
                  const torch::Tensor *target_nodes,
                  const double p,
                  const double q,
                  const int walk_length,
                  const int seed)
{

  if(row_ptr->device().is_cuda()) {
    return walk_gpu(row_ptr,column_idx,target_nodes,p,q,walk_length,seed);
  }else{
    return walk_cpu(row_ptr,column_idx,target_nodes,p,q,walk_length,seed);
  }
}

torch::Tensor walk_edge_list(const torch::Tensor *edge_list,
                  const torch::Tensor *node_edges_idx,
                  const torch::Tensor *target_nodes,
                  const double p,
                  const double q,
                  const int walk_length,
                  const int seed)
{

  if(row_ptr->device().is_cuda()) {
    throw;
  }else{
    return walk_edge_list_cpu(edge_list,node_edges_idx,target_nodes,p,q,walk_length,seed);
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> to_windows(const torch::Tensor *walks,
                                      const int window_size,
                                      const int64_t num_nodes,
                                      const int seed
                                    )
{
  if(walks->device().is_cuda()) {
    return to_windows_gpu(walks,window_size,num_nodes,seed);
  }else{
    return to_windows_cpu(walks,window_size,num_nodes,seed);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("walk", &walk, "walk");
  m.def("walk_edge_list", &walk_edge_list, "walk_edge_list");
  m.def("to_windows",&to_windows,"to_windows");
}
