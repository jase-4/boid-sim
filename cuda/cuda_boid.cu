#include "cuda_boid.cuh"
#include <cstdio> 

__constant__  BoidParams d_params;
__device__ float group_bias_sum[MAX_GROUPS];
__device__ int group_bias_count[MAX_GROUPS];


__host__ __device__ inline float clamp(float x, float minVal, float maxVal) {
    return fminf(fmaxf(x, minVal), maxVal);
}


__global__ void boid_update_kernel(Vec3* positions, Vec3* velocities, int* group_ids, GroupParams* group_params,  int count, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    Vec3 pos_avg(0.0f, 0.0f, 0.0f), velo_avg(0.0f, 0.0f, 0.0f), close(0.0f, 0.0f, 0.0f);
    float neighbors = 0.0f;
    Vec3 self_pos = positions[i];

    for (int j = 0; j < count; ++j) {
        if (i == j) continue;
        Vec3 offset = self_pos - positions[j];
        float dist_sq = offset.length_squared();

        if (dist_sq < d_params.visual_range_sq) {
            if (dist_sq < d_params.protected_range_sq)
                close += offset;
            else {
                pos_avg += positions[j];
                velo_avg += velocities[j];
                neighbors += 1.0f;
            }
        }
    }

    Vec3 v = velocities[i];
    if (neighbors > 0.0f) {
        pos_avg = pos_avg * (1.0f / neighbors);
        velo_avg = velo_avg * (1.0f / neighbors);
        v += (pos_avg - self_pos) * d_params.centering_factor;
        v += (velo_avg - v) * d_params.matching_factor;
    }

    v += close * d_params.avoid_factor;

    //change to global eventually
    Vec3 box_min(0.0f, -20.0f, 0.0f), box_max(50.0f, 100.0f, 50.0f);
    keep_in_bounds(self_pos, v, box_min, box_max, d_params.margin);


   


    int gid = group_ids[i];
     if (gid < 0 || gid >= MAX_GROUPS) {
         printf("Invalid gid: %d at index %d\n", gid, i);
        return;
    }
if (gid > 0 && gid < MAX_GROUPS) {
    Vec3 dir = group_params[gid].direction;
    float dot = v.dot(dir);

    float bias_inc = (dot > 0.0f) ? d_params.bias_increment : -d_params.bias_increment;

    atomicAdd(&group_bias_sum[gid], bias_inc);
    atomicAdd(&group_bias_count[gid], 1);
}



    float speed = v.length();
    if (speed < d_params.min_speed)
        v = v.normalize() * d_params.min_speed;
    else if (speed > d_params.max_speed)
        v = v.normalize() * d_params.max_speed;
     
 
    positions[i] += v * dt;
    velocities[i] = v;
}

__global__ void update_group_bias(float max_bias, GroupParams* group_params ) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= MAX_GROUPS - 1) return;

    int count = group_bias_count[gid];
    if (count > 0) {
        float avg = group_bias_sum[gid] / count;
        float new_bias = group_params[gid].bias_val + avg;
        group_params[gid].bias_val = clamp(new_bias, 0.0f, max_bias);
    }
    group_bias_sum[gid] = 0.0f;
    group_bias_count[gid] = 0;
}



__device__ void keep_in_bounds(Vec3& pos, Vec3& vel, const Vec3& box_min, const Vec3& box_max, float margin) {
    // X-axis
    if (pos.x < box_min.x + margin) {
        pos.x = box_min.x + margin;
        vel.x = fabsf(vel.x);
    } else if (pos.x > box_max.x - margin) {
        pos.x = box_max.x - margin;
        vel.x = -fabsf(vel.x);
    }

    // Y-axis
    if (pos.y < box_min.y + margin) {
        pos.y = box_min.y + margin;
        vel.y = fabsf(vel.y);
    } else if (pos.y > box_max.y - margin) {
        pos.y = box_max.y - margin;
        vel.y = -fabsf(vel.y);
    }

    // Z-axis
    if (pos.z < box_min.z + margin) {
        pos.z = box_min.z + margin;
        vel.z = fabsf(vel.z);
    } else if (pos.z > box_max.z - margin) {
        pos.z = box_max.z - margin;
        vel.z = -fabsf(vel.z);
    }
}
