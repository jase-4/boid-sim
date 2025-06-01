
#include "boid_params.hpp"
//include "symbol_defs.cuh"
#include "boid_sim_cuda.hpp"
#include "vec3.cuh"
#include <glm/glm.hpp>
#include "cuda_boid.cuh"
#include "constants.hpp"
#include <iostream>




inline Vec3 toVec3(const glm::vec3& v) {
    return Vec3(v.x, v.y, v.z);
}

inline glm::vec3 toGlmVec3(const Vec3& v) {
    return glm::vec3(v.x, v.y, v.z);
}
//__constant__ BoidParams d_params;


void BoidSimCUDA::init_boid_params() {
     BoidParams params = {
        1.0f,                       // protected_range_sq
        25.0f,                       // visual_range_sq
        0.0005f,                    // centering_factor
        0.05f,                      // matching_factor
        0.05f,                      // avoid_factor
        4.0f,                      // min_speed
        10.0f,                       // max_speed
        0.5f,                       // margin
         0.01f,                     // bias_increment
        // {0.0f, 0.0f, 0.0f},         // box_min (brace-initialized Vec3)
        // {100.0f, 100.0f, 100.0f}    // box_max/ box_max
    };

    cudaMemcpyToSymbol(d_params, &params, sizeof(BoidParams));
    cudaError_t err = cudaMemcpyToSymbol(d_params, &params, sizeof(BoidParams));
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpyToSymbol failed: " << cudaGetErrorString(err) << std::endl;
    }
}



 void BoidSimCUDA::init_group_directions() {
    group_params.resize(MAX_GROUPS);  // Ensure the vector has MAX_GROUPS elements
    for (int i = 0; i < MAX_GROUPS; ++i) {
        group_params[i].direction = Vec3(1.0f, 0.0f, 0.0f);  // example init
        group_params[i].bias_val = 0.0f;
    }

   
    float zeros[MAX_GROUPS] = {0};
    int izeros[MAX_GROUPS] = {0};

    cudaMemcpyToSymbol(group_bias_sum, zeros, sizeof(float) * MAX_GROUPS);
    cudaMemcpyToSymbol(group_bias_count, izeros, sizeof(int) * MAX_GROUPS);
}



void BoidSimCUDA::updateBoidsCUDA(float dt) {
  
    Vec3* positions = new Vec3[count];
    Vec3* velocities = new Vec3[count];
    int* groups = new int[count];
  
    for (int i = 0; i < count; ++i) {
        positions[i] = toVec3(positions_glm[i]);
        velocities[i] = toVec3(velocities_glm[i]);
        groups[i] = group_ids[i];
    }

    Vec3 *d_positions, *d_velocities;
    int* d_group_ids;
    GroupParams* d_group_params;

    size_t vec_size = count * sizeof(Vec3);
    size_t group_size = count * sizeof(int);

    cudaMalloc(&d_positions, vec_size);
    cudaMalloc(&d_velocities, vec_size);
    cudaMalloc(&d_group_ids, group_size);

cudaMalloc(&d_group_params, group_params.size() * sizeof(GroupParams));

    cudaMemcpy(d_positions, positions, vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, velocities, vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_group_ids, groups, group_size, cudaMemcpyHostToDevice);
cudaMemcpy(d_group_params, group_params.data(), group_params.size() * sizeof(GroupParams), cudaMemcpyHostToDevice);


   
    int threadsPerBlock = 256;
    int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;
  
    boid_update_kernel<<<blocks, threadsPerBlock>>>(d_positions, d_velocities, d_group_ids, d_group_params, count, dt);
    cudaDeviceSynchronize();

    
    int groupThreads = 64;
    int groupBlocks = (MAX_GROUPS + groupThreads - 1) / groupThreads;
   update_group_bias<<<groupBlocks, groupThreads>>>(0.03f,d_group_params); // max_bias = 1.0f
    cudaDeviceSynchronize();

    cudaMemcpy(positions, d_positions, vec_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities, d_velocities, vec_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(groups, d_group_ids, group_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(group_params.data(), d_group_params, group_params.size(), cudaMemcpyDeviceToHost);

    for (int i = 0; i < count; ++i) {
        positions_glm[i] = toGlmVec3(positions[i]);
        velocities_glm[i] = toGlmVec3(velocities[i]);
        group_ids[i] = groups[i];
    }

    // Cleanup
    delete[] positions;
    delete[] velocities;
    delete[] groups;
    cudaFree(d_positions);
    cudaFree(d_velocities);
    cudaFree(d_group_ids);
    cudaFree(d_group_params);
}
