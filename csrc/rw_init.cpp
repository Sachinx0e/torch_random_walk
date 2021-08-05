#include <torch/extension.h>
#include "cpu/rw_cpu.h"
#include "cuda/rw_cuda.h"

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("walk", &walk, "walk");
}