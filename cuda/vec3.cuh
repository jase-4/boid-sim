#ifndef VEC3_CUH
#define VEC3_CUH

#include <cuda_runtime.h>



struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__ Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }

    __host__ __device__ Vec3 operator*(float s) const {
        return Vec3(x * s, y * s, z * s);
    }

    __host__ __device__ Vec3& operator+=(const Vec3& v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }
    __host__ __device__ Vec3& operator-=(const Vec3& v) {
        x -= v.x; y -= v.y; z -= v.z;
        return *this;
    }


    __host__ __device__ float length_squared() const {
        return x * x + y * y + z * z;
    }

    __host__ __device__ float length() const {
        return sqrtf(length_squared());
    }

    __host__ __device__ Vec3 normalize() const {
        float len = length();
        return len > 0 ? (*this) * (1.0f / len) : Vec3(0, 0, 0);
    }

   __host__ __device__ float dot( const Vec3& a) {
    return a.x * (*this).x + a.y * (*this).y + a.z * (*this).z;
}

};



#endif
