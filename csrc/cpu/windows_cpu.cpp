#include "windows_cpu.h"
#include "../cuda/utils.cuh"
#include "cpu_utils.h"

std::tuple<at::Tensor, at::Tensor, at::Tensor> to_windows_cpu(const torch::Tensor *walks,
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
        
    return std::make_tuple(target_nodes,pos_windows,neg_windows);
}


std::tuple<at::Tensor, at::Tensor, at::Tensor> to_windows_cbow_cpu(const torch::Tensor *walks,
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
    auto pos_nodes = torch::empty({num_windows},torch::kInt64);
    auto neg_nodes = torch::empty({num_windows},torch::kInt64);
    auto windows = torch::empty({num_windows,window_size-1},torch::kInt64);

    // grain size
    int grain_size = torch::internal::GRAIN_SIZE / num_walks;

    // create accessors
    auto walks_accessor = walks->accessor<int64_t,2>();
    auto pos_nodes_accessor = pos_nodes.accessor<int64_t,1>();
    auto neg_nodes_accesor = neg_nodes.accessor<int64_t,1>();
    auto windows_accesor = windows.accessor<int64_t,2>();

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
                auto pos_node = walk[target_node_idx];
                pos_nodes_accessor[target_node_pos] = pos_node;
                
                // sample negative node
                auto neg_node = sample_int(0,(num_nodes-1));
                auto max_checks = 0;
                while(neg_node == pos_node && max_checks <= 100){
                    neg_node = sample_int(0,(num_nodes-1));
                    max_checks = max_checks + 1;
                }
                
                neg_nodes_accesor[target_node_pos] = neg_node;

                // create pos window
                auto pos_window = windows_accesor[target_node_pos];
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
                                
                
                
            }
        }
    });
        
    return std::make_tuple(pos_nodes,neg_nodes,windows);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> to_windows_triples_cpu(const torch::Tensor *walks,
                        const int window_size,
                        const int64_t num_nodes,
                        const int64_t padding_idx,
                        const torch::Tensor *triples,
                        const int seed
                    ){

    // seed
    srand(seed);

    // check walks is contiguous
    CHECK_CONTIGUOUS(walks);

    // calculate sizes
    int64_t num_walks = walks->size(0);
    int64_t walk_length = walks->size(1);
    int64_t num_windows_in_one_walk = (walk_length - 1)/2; 
    int64_t num_windows_for_all_walks = num_windows_in_one_walk*num_walks;
    int64_t num_triples = triples->size(0);

    // create arrays to hold results
    auto target_triples = torch::empty({num_windows_for_all_walks,3},torch::kInt64);
    auto pos_windows = torch::empty({num_windows_for_all_walks,window_size*2,3},torch::kInt64);
    auto neg_windows = torch::empty({num_windows_for_all_walks,window_size*2,3},torch::kInt64);

    // grain size
    int grain_size = torch::internal::GRAIN_SIZE / num_walks;

    // create accessors
    auto walks_accessor = walks->accessor<int64_t,2>();
    auto target_triples_accessor = target_triples.accessor<int64_t,2>();
    auto pos_windows_accesor = pos_windows.accessor<int64_t,3>();
    auto neg_windows_accesor = neg_windows.accessor<int64_t,3>();
    auto triples_accesor = triples->accessor<int64_t,2>();

    // do work
    torch::parallel_for(0,num_walks,grain_size,[&](int64_t walk_idx_start,int64_t walk_idx_end){
        
        // loop over walks
        for (int64_t walk_idx = 0;walk_idx < num_walks;walk_idx++){
            
            // get the walk at this index
            auto walk = walks_accessor[walk_idx];

            
            // hop by two steps
            int64_t target_index = 0;
            for(int64_t target_rel_index=1;target_rel_index<walk_length-1;target_rel_index+=2){
                                
                // calculate the position in target triples tensor
                auto target_pos = (num_windows_in_one_walk * walk_idx) + target_index;


                // get the target triple
                target_triples_accessor[target_pos][0] = walk[target_rel_index -1];     // Head
                target_triples_accessor[target_pos][1] = walk[target_rel_index];         // Relation
                target_triples_accessor[target_pos][2] = walk[target_rel_index + 1];     // Tail

                // get left positive windows
                for(int64_t hop=0;hop<=window_size;hop++){
                                
                    // get head
                    auto rel_idx = target_rel_index - ((hop + 1)*2);
                    auto head_idx = rel_idx - 1;
                    auto tail_idx = rel_idx + 1;
                                         
                    // head node
                    if(head_idx >= 0){
                        pos_windows_accesor[target_pos][hop][0] = walk[rel_idx];;
                    }else{
                        pos_windows_accesor[target_pos][hop][0] = padding_idx;
                    }

                    // rel node
                    if(rel_idx >= 0){
                        pos_windows_accesor[target_pos][hop][1] = walk[rel_idx];
                    }else{
                        pos_windows_accesor[target_pos][hop][1] = padding_idx;
                    }

                    // tail node
                    if(tail_idx >= 0){
                        pos_windows_accesor[target_pos][hop][2] = walk[tail_idx];
                    }else{
                        pos_windows_accesor[target_pos][hop][2] = padding_idx;
                    }

                }

                // get right positive windows
                for(int64_t hop=0;hop<window_size;hop++){
                                
                    // window item position
                    auto window_item_pos = hop+window_size;

                    // get head
                    auto rel_idx = target_rel_index + ((hop + 1)*2);
                    auto head_idx = rel_idx - 1;
                    auto tail_idx = rel_idx + 1;

                    //printf("Hop index: %d / Target rel index: %d / Rel index: %d \n",hop,target_rel_index, rel_idx);                    
                                         
                    // head node
                    if(head_idx < walk_length){
                        pos_windows_accesor[target_pos][window_item_pos][0] = walk[head_idx];;
                    }else{
                        pos_windows_accesor[target_pos][window_item_pos][0] = padding_idx;
                    }

                    // rel node
                    if(rel_idx < walk_length){
                        pos_windows_accesor[target_pos][window_item_pos][1] = walk[rel_idx];
                    }else{
                        pos_windows_accesor[target_pos][window_item_pos][1] = padding_idx;
                    }

                    // tail node
                    if(tail_idx < walk_length){
                        pos_windows_accesor[target_pos][window_item_pos][2] = walk[tail_idx];
                    }else{
                        pos_windows_accesor[target_pos][window_item_pos][2] = padding_idx;
                    }

                }
                

                // create negatives
                for(int64_t hop=0;hop<(window_size*2);hop++){
                    
                    auto triple_idx = sample_int(0,(num_triples-1));

                    auto triple = triples_accesor[triple_idx];

                    // decide what to corrupt
                    neg_windows_accesor[target_pos][hop][0] = triple[0];
                    neg_windows_accesor[target_pos][hop][1] = triple[1];
                    neg_windows_accesor[target_pos][hop][2] = triple[2];
                
                }

                // update target index
                target_index = target_index + 1;

            }
        }
    });
        
    return std::make_tuple(target_triples,pos_windows,neg_windows);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> to_windows_triples_cbow_cpu(const torch::Tensor *walks,
                        const int window_size,
                        const int64_t num_nodes,
                        const int64_t padding_idx,
                        const torch::Tensor *triples,
                        const int seed
                    ){

    // seed
    srand(seed);

    // check walks is contiguous
    CHECK_CONTIGUOUS(walks);

    // calculate sizes
    int64_t num_walks = walks->size(0);
    int64_t walk_length = walks->size(1);
    int64_t num_windows_in_one_walk = (walk_length - 1)/2; 
    int64_t num_windows_for_all_walks = num_windows_in_one_walk*num_walks;
    int64_t num_triples = triples->size(0);

    // create arrays to hold results
    auto pos_triples = torch::empty({num_windows_for_all_walks,3},torch::kInt64);
    auto pos_windows = torch::empty({num_windows_for_all_walks,window_size*2,3},torch::kInt64);
    auto neg_triples = torch::empty({num_windows_for_all_walks,3},torch::kInt64);

    // grain size
    int grain_size = torch::internal::GRAIN_SIZE / num_walks;

    // create accessors
    auto walks_accessor = walks->accessor<int64_t,2>();
    auto pos_triples_accessor = pos_triples.accessor<int64_t,2>();
    auto neg_triples_accesor = neg_triples.accessor<int64_t,2>();
    auto pos_windows_accesor = pos_windows.accessor<int64_t,3>();
    auto triples_accesor = triples->accessor<int64_t,2>();

    // do work
    torch::parallel_for(0,num_walks,grain_size,[&](int64_t walk_idx_start,int64_t walk_idx_end){
        
        // loop over walks
        for (int64_t walk_idx = 0;walk_idx < num_walks;walk_idx++){
            
            // get the walk at this index
            auto walk = walks_accessor[walk_idx];

            
            // hop by two steps
            int64_t target_index = 0;
            for(int64_t target_rel_index=1;target_rel_index<walk_length-1;target_rel_index+=2){
                                
                // calculate the position in target triples tensor
                auto target_pos = (num_windows_in_one_walk * walk_idx) + target_index;


                // get the target triple
                auto pos_head = walk[target_rel_index -1];
                auto pos_rel = walk[target_rel_index]; 
                auto pos_tail = walk[target_rel_index + 1];
                
                pos_triples_accessor[target_pos][0] = pos_head;     // Head
                pos_triples_accessor[target_pos][1] = pos_rel;       // Relation
                pos_triples_accessor[target_pos][2] = pos_tail;     // Tail

                // sample negative triple
                auto neg_triple_idx = sample_int(0,(num_triples-1));
                auto neg_head = triples_accesor[neg_triple_idx][0];
                auto neg_rel = triples_accesor[neg_triple_idx][1];
                auto neg_tail = triples_accesor[neg_triple_idx][2];

                // loop until we find a negative triple that is not equal to positive triple
                auto max_checks = 0;
                while(neg_head == pos_head && neg_rel == pos_rel && neg_tail == pos_tail && max_checks<=100){
                    // sample negative triple
                    neg_triple_idx = sample_int(0,(num_triples-1));
                    neg_head = triples_accesor[neg_triple_idx][0];
                    neg_rel = triples_accesor[neg_triple_idx][1];
                    neg_tail = triples_accesor[neg_triple_idx][2];

                    max_checks = max_checks + 1;
                }

                neg_triples_accesor[target_pos][0] = neg_head;
                neg_triples_accesor[target_pos][1] = neg_rel;
                neg_triples_accesor[target_pos][2] = neg_tail;

                // get left positive windows
                for(int64_t hop=0;hop<=window_size;hop++){
                                
                    // get head
                    auto rel_idx = target_rel_index - ((hop + 1)*2);
                    auto head_idx = rel_idx - 1;
                    auto tail_idx = rel_idx + 1;
                                         
                    // head node
                    if(head_idx >= 0){
                        pos_windows_accesor[target_pos][hop][0] = walk[rel_idx];;
                    }else{
                        pos_windows_accesor[target_pos][hop][0] = padding_idx;
                    }

                    // rel node
                    if(rel_idx >= 0){
                        pos_windows_accesor[target_pos][hop][1] = walk[rel_idx];
                    }else{
                        pos_windows_accesor[target_pos][hop][1] = padding_idx;
                    }

                    // tail node
                    if(tail_idx >= 0){
                        pos_windows_accesor[target_pos][hop][2] = walk[tail_idx];
                    }else{
                        pos_windows_accesor[target_pos][hop][2] = padding_idx;
                    }

                }

                

                // get right positive windows
                for(int64_t hop=0;hop<window_size;hop++){
                                
                    // window item position
                    auto window_item_pos = hop+window_size;

                    // get head
                    auto rel_idx = target_rel_index + ((hop + 1)*2);
                    auto head_idx = rel_idx - 1;
                    auto tail_idx = rel_idx + 1;

                    //printf("Hop index: %d / Target rel index: %d / Rel index: %d \n",hop,target_rel_index, rel_idx);                    
                                         
                    // head node
                    if(head_idx < walk_length){
                        pos_windows_accesor[target_pos][window_item_pos][0] = walk[head_idx];;
                    }else{
                        pos_windows_accesor[target_pos][window_item_pos][0] = padding_idx;
                    }

                    // rel node
                    if(rel_idx < walk_length){
                        pos_windows_accesor[target_pos][window_item_pos][1] = walk[rel_idx];
                    }else{
                        pos_windows_accesor[target_pos][window_item_pos][1] = padding_idx;
                    }

                    // tail node
                    if(tail_idx < walk_length){
                        pos_windows_accesor[target_pos][window_item_pos][2] = walk[tail_idx];
                    }else{
                        pos_windows_accesor[target_pos][window_item_pos][2] = padding_idx;
                    }

                }
                

                // update target index
                target_index = target_index + 1;

            }
        }
    });
        
    return std::make_tuple(pos_triples,neg_triples,pos_windows);
}