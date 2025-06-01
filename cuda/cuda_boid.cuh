#ifndef CUDA_BOID_CUH
#define CUDA_BOID_CUH

#include "cuda_runtime.h"
#include "vec3.cuh"
#include "boid_params.hpp"
#include "symbol_defs.cuh"
#include "group_params.hpp"



extern __device__ float group_bias_sum[MAX_GROUPS];
extern __device__ int group_bias_count[MAX_GROUPS];




__global__ void update_group_bias(float max_bias, GroupParams* group_params);
__device__ void keep_in_bounds(Vec3& pos, Vec3& vel, const Vec3& box_min, const Vec3& box_max, float margin);
__global__ void boid_update_kernel(Vec3* positions, Vec3* velocities, int* group_ids, GroupParams* group_params,  int count, float dt);
 
#endif