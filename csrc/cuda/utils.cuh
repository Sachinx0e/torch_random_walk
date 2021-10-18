#pragma once

#include <torch/extension.h>
#include <curand.h>
#include <curand_kernel.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(x.is_cpu(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x->is_contiguous(), #x " must be a contigous tensor")

__device__ inline int64_t sample_int_gpu(int64_t start, int64_t end,curandState_t* rand_state){
    auto sampled_int = start + (curand(rand_state) % (((end + 1) - start)));
    return sampled_int;
}