#include <torch/extension.h>
#include "cpu/rw_cpu.h"
#include "cpu/rw_cpu_edge_list.h"
#include "cuda/rw_cuda_edge_list.h"
#include "cuda/rw_cuda.h"
#include "cpu/windows_cpu.h"
#include "cuda/windows_cuda.h"
#include "cpu/rw_cpu_triples.h"
#include "cuda/rw_cuda_triples.h"

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

torch::Tensor walk_edge_list(const torch::Tensor *edge_list_indexed,
                  const torch::Tensor *node_edges_idx,
                  const torch::Tensor *target_nodes,
                  const double p,
                  const double q,
                  const int walk_length,
                  const int seed,
                  const int64_t padding_idx,
                  const bool restart
                )
{

  if(edge_list_indexed->device().is_cuda()) {
    return walk_edge_list_gpu(edge_list_indexed,node_edges_idx,target_nodes,p,q,walk_length,seed,padding_idx,restart);
  }else{
    return walk_edge_list_cpu(edge_list_indexed,node_edges_idx,target_nodes,p,q,walk_length,seed,padding_idx,restart);
  }

}

torch::Tensor walk_triples(const torch::Tensor *triples_indexed,
                  const torch::Tensor *relation_tail_index,
                  const torch::Tensor *target_nodes,
                  const int walk_length,
                  const int64_t padding_idx,
                  const bool restart,
                  const int seed
                )
{

  if(target_nodes->device().is_cuda()) {
    return triples::walk_triples_gpu(triples_indexed,
                                    relation_tail_index,
                                    target_nodes,
                                    walk_length,
                                    padding_idx,
                                    restart,
                                    seed);
  }else{
    return triples::walk_triples_cpu(triples_indexed,
                                     relation_tail_index,
                                     target_nodes,
                                     walk_length,
                                     padding_idx,
                                     restart,
                                     seed);
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> to_windows_triples(const torch::Tensor *walks,
                                      const int window_size,
                                      const int64_t num_nodes,
                                      const int64_t padding_idx,
                                      const torch::Tensor *triples,
                                      const int seed
                                    )
{
  if(walks->device().is_cuda()) {
    throw;
  }else{
    return to_windows_triples_cpu(walks,window_size,num_nodes,padding_idx,triples,seed);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("walk", &walk, "walk");
  m.def("walk_edge_list", &walk_edge_list, "walk_edge_list");
  m.def("walk_triples", &walk_triples, "walk_triples");
  m.def("to_windows",&to_windows,"to_windows");
  m.def("to_windows_triples",&to_windows_triples,"to_windows_triples");
}
