#ifndef GROUP_PARAMS_HPP
#define GROUP_PARAMS_HPP

#include "vec3.cuh"



struct GroupParams {
    Vec3 direction;
    float bias_val;

    __host__ __device__
    GroupParams() : direction(0.0f, 0.0f, 0.0f), bias_val(0.0f) {}

    __host__ __device__
    GroupParams(const Vec3& dir, float bias) : direction(dir), bias_val(bias) {}
};

#endif