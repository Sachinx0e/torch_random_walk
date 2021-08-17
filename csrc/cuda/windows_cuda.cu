#include "windows_cuda.h"
#include "utils.cuh"
#include <curand.h>
#include <curand_kernel.h>

__global__ void create_windows(torch::PackedTensorAccessor64<int64_t,2> walks_accessor,
                               const int num_walks,
                               const int walk_length,
                               const int window_size,
                               const int mid_pos,
                               const int64_t num_nodes, 
                               torch::PackedTensorAccessor64<int64_t,1> target_nodes_accessor,
                               torch::PackedTensorAccessor64<int64_t,2> pos_windows_accesor,
                               torch::PackedTensorAccessor64<int64_t,2> neg_windows_accesor,
                               const int seed 
                            )
{

    // get the thread
    const auto thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    // seed rng
    curandState_t rand_state;
    curand_init(seed,thread_index,1,&rand_state);

    // check bounds
    if(thread_index < num_walks){
        auto walk_idx  = thread_index;

        // get the walk for this index
        auto walk = walks_accessor[walk_idx];

        // loop over this walk
        auto step_end = (walk_length - window_size) + 1;
        for(int64_t step_idx=0;step_idx<step_end;step_idx++){
            auto window_start = step_idx;
                        
            // calculate position in target nodes
            int64_t target_node_pos = (walk_idx * step_end) + step_idx;
            int64_t target_node_idx = window_start+mid_pos;
            target_nodes_accessor[target_node_pos] = walk[target_node_idx];
            
            // create pos window
            auto pos_window = pos_windows_accesor[target_node_pos];
            int64_t pos_index = 0;
            for(int i = 0;i<window_size;i++){
                auto walk_pos = window_start+i;
                if(i != mid_pos){
                    pos_window[pos_index] = walk[walk_pos];
                    pos_index = pos_index + 1;
                }else{
                    pos_index = pos_index;
                }    
            }
                            
            // create negative window
            auto neg_windows = neg_windows_accesor[target_node_pos]; 
            for(int i = 0;i<window_size-1;i++){
                auto nbr_node = 0 + (curand(&rand_state) % ((num_nodes) - 0));
                neg_windows[i] = nbr_node;
            }
        }
    }
}

std::vector<torch::Tensor> to_windows_gpu(const torch::Tensor *walks,
                        const int window_size,
                        const int64_t num_nodes,
                        const int seed
                    ){

    // check walks is contiguous
    CHECK_CUDA((*walks));
    CHECK_CONTIGUOUS(walks);

    cudaSetDevice(walks->device().index());

    // calculate sizes
    int64_t num_walks = walks->size(0);
    int64_t walk_length = walks->size(1);
    int64_t num_windows = ((walk_length - window_size)+1)*num_walks;
    int64_t mid_pos = int64_t(window_size/2);

    // create arrays to hold results
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA,walks->device().index());  
    auto target_nodes = torch::empty({num_windows},options);
    auto pos_windows = torch::empty({num_windows,window_size-1},options);
    auto neg_windows = torch::empty({num_windows,window_size-1},options);

    // create accessors
    auto walks_accessor = walks->packed_accessor64<int64_t,2>();
    auto target_nodes_accessor = target_nodes.packed_accessor64<int64_t,1>();
    auto pos_windows_accesor = pos_windows.packed_accessor64<int64_t,2>();
    auto neg_windows_accesor = neg_windows.packed_accessor64<int64_t,2>();

    // Thread block size
    int NUM_THREADS = 1024;

    // Grid size
    int NUM_BLOCKS = int((num_walks + NUM_THREADS - 1)/NUM_THREADS);
    
    // launch kernel
    create_windows<<<NUM_BLOCKS,NUM_THREADS>>>(walks_accessor,
                                            num_walks,
                                            walk_length,
                                            window_size,
                                            mid_pos,
                                            num_nodes,
                                            target_nodes_accessor,
                                            pos_windows_accesor,
                                            neg_windows_accesor,
                                            seed
                                        );
    
    return {target_nodes,pos_windows,neg_windows};
}