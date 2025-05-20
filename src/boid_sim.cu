#include "boid_sim.cuh"


__global__ void updateBoidsKernel(glm::vec3* positions, glm::vec3* velocities, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    Vec3 duh = Vec3(0.0f,0.0f,0.0f);

    // Naive update, e.g., move boid forward
    velocities[idx] += glm::vec3(0.01f, 0.0f, 0.0f);  // acceleration
    positions[idx] += velocities[idx] * 0.001f;                // position update
}