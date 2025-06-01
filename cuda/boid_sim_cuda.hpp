#ifndef BOID_SIM_CUDA
#define BOID_SIM_CUDA

#include <glm/glm.hpp>
#include <vector>
#include "group_params.hpp"
#include "constants.hpp"

class BoidSimCUDA{
    public:
    void init_boid_params();
    void updateBoidsCUDA(float dt);
    void init_group_directions();
    void init_group_ids(int base_group_size);
   
   std::vector<glm::vec3> positions_glm;
    std::vector<glm::vec3> velocities_glm;
     std::vector<int> group_ids;
     std::vector<GroupParams> group_params;

    int count;
};


#endif