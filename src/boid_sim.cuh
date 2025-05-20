#include "vec3.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
__global__ void updateBoidsKernel(glm::vec3* positions, glm::vec3* velocities, int count);