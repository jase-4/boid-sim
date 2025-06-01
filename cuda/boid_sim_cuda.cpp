#include "boid_sim_cuda.hpp"


void BoidSimCUDA::init_group_ids(int base_group_size) {
    group_ids.assign(count, 0); 

    int actual_groups = MAX_GROUPS - 1;
    int total_assigned = actual_groups * base_group_size;

    if (total_assigned > count) {
       
        return;
    }

    int index = 0;

   
    for (int group = 1; group <= actual_groups; ++group) {
        int assigned = 0;
        while (assigned < base_group_size && index < count) {
            
            if (group_ids[index] == 0) {
                group_ids[index] = group;
                ++assigned;
            }
            ++index;
        }
    }
}