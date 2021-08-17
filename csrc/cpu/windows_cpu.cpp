#include "windows_cpu.h"
#include "../cuda/utils.cuh"

std::vector<torch::Tensor> to_windows_cpu(const torch::Tensor *walks,
                        const int window_size,
                        const int64_t num_nodes,
                        const int seed
                        ){

    // seed
    srand(seed);

    // check walks is contiguous
    CHECK_CONTIGUOUS(walks);

    // calculate sizes
    int64_t num_walks = walks->size(0);
    int64_t walk_length = walks->size(1);
    int64_t num_windows = ((walk_length - window_size)+1)*num_walks;
    int64_t mid_pos = int64_t(window_size/2);

    // create arrays to hold results
    auto target_nodes = torch::empty({num_windows},torch::kInt64);
    auto pos_windows = torch::empty({num_windows,window_size-1},torch::kInt64);
    auto neg_windows = torch::empty({num_windows,window_size-1},torch::kInt64);

    // grain size
    int grain_size = torch::internal::GRAIN_SIZE / num_walks;

    // create accessors
    auto walks_accessor = walks->accessor<int64_t,2>();
    auto target_nodes_accessor = target_nodes.accessor<int64_t,1>();
    auto pos_windows_accesor = pos_windows.accessor<int64_t,2>();
    auto neg_windows_accesor = neg_windows.accessor<int64_t,2>();

    // do work
    torch::parallel_for(0,num_walks,grain_size,[&](int64_t walk_idx_start,int64_t walk_idx_end){
        for (int64_t walk_idx = 0;walk_idx < num_walks;walk_idx++){
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
                    auto nbr_node = 0 + ( std::rand() % ( (num_nodes) - 0));
                    neg_windows[i] = nbr_node;
                }
            }
        }
    });
        
    return {target_nodes,pos_windows,neg_windows};
}