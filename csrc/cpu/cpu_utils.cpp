#include <random>

int64_t sample_int(int64_t start, int64_t end){
   if(start == end){
       return start;
   }else{
       auto sampled_int = start + ( std::rand() % ((end +1) - start));
       return sampled_int; 
   }
}