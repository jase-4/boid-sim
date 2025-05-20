#ifndef BOID_SIM_HPP
#define BOID_SIM_HPP


#include <glm/glm.hpp>

extern "C" void updateBoidsCUDA(glm::vec3* positions, glm::vec3* velocities, int count);
#endif