#pragma once

#include "cuda_runtime.h"
#include <GPUMCI/physics/CudaSettings.h>

// Only include STL headers if this isn't being processed by the CUDA compiler
#include <vector>

namespace gpumci {
namespace cuda {

// Reference used: https://forums.developer.nvidia.com/t/cuda-for-quaternions-hyper-complex-numbers-operations/44116/2

struct Quaternion {
    // Default constructor
    __device__ Quaternion() : q(make_float4(1.0f, 0.0f, 0.0f, 0.0f)) {}

    // Constructor with given components
    __device__ Quaternion(float w, float x, float y, float z) : q(make_float4(w, x, y, z)) {}

    static __device__ __forceinline__ Quaternion describe_rotation(const float3 v, const float sina_2, const float cosa_2) {
        Quaternion result;
        result.q = make_float4(cosa_2, sina_2 * v.x, sina_2 * v.y, sina_2 * v.z);
        return result;
    }

    static __device__ __forceinline__ Quaternion describe_rotation(const float3 v, const float angle) {
        float sina_2 = sinf(angle / 2);
        float cosa_2 = cosf(angle / 2);
        Quaternion result;
        result.q = make_float4(cosa_2, sina_2 * v.x, sina_2 * v.y, sina_2 * v.z);
        return result;
    }

    __device__ __forceinline__ float3 rotate(const float3 v) const {
        float t2 = q.x * q.y;
        float t3 = q.x * q.z;
        float t4 = q.x * q.w;
        float t5 = -q.y * q.y;
        float t6 = q.y * q.z;
        float t7 = q.y * q.w;
        float t8 = -q.z * q.z;
        float t9 = q.z * q.w;
        float t10 = -q.w * q.w;
        return make_float3(
            2.0f * ((t8 + t10) * v.x + (t6 - t4) * v.y + (t3 + t7) * v.z) + v.x,
            2.0f * ((t4 + t6) * v.x + (t5 + t10) * v.y + (t9 - t2) * v.z) + v.y,
            2.0f * ((t7 - t3) * v.x + (t2 + t9) * v.y + (t5 + t8) * v.z) + v.z);
    }

    __device__ __forceinline__ float3 rotate_around_p(const float3 v, const float3 p) const {
        return p + rotate(v - p);
    }

  protected:
    // 1,i,j,k
    float4 q;

    // Kernel function to rotate particles
    //__global__ void cudaRotationKernel(cuda::CudaMonteCarloParticle* particles, int numParticles, float angle, float3 rotationAxis);
};

__global__ void cudaRotationKernel(CudaMonteCarloParticle* particles, int numParticles, float angle, float3 rotationAxis);

void rotateParticles(std::vector<cuda::CudaMonteCarloParticle>& particles, float3& axis, float angle);

} // namespace cuda
} // namespace gpumci
