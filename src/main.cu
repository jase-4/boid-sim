#include "boid_sim.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>


extern "C" void updateBoidsCUDA(glm::vec3* positions, glm::vec3* velocities, int count) {
    glm::vec3* d_positions;
    glm::vec3* d_velocities;

    size_t size = count * sizeof(glm::vec3);
    cudaMalloc(&d_positions, size);
    cudaMalloc(&d_velocities, size);

    cudaMemcpy(d_positions, positions, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, velocities, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;
    updateBoidsKernel<<<blocks, threadsPerBlock>>>(d_positions, d_velocities, count);
    cudaDeviceSynchronize();

    cudaMemcpy(positions, d_positions, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities, d_velocities, size, cudaMemcpyDeviceToHost);

    cudaFree(d_positions);
    cudaFree(d_velocities);
}
