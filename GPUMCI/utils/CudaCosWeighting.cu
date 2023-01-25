#include <vector>
#include <iostream>
#include <cassert>

#include <float.h>
#include <cstdio>
#include <odl_cpp_utils/cuda/cutil_math.h>
#include <odl_cpp_utils/cuda/cuda_utils.h>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// thrust
#include <thrust/device_vector.h>
#include <odl_cpp_utils/cuda/texture.h>

namespace gpumci {
namespace cuda {

__global__ void cosWeightingKernel(const float3 sourcePosition,
                                   const float3 detectorOrigin,
                                   const float3 pixelDirectionU,
                                   const float3 pixelDirectionV,
                                   const int2 detectorSize,
                                   float* source,
                                   float* target) {
    const uint2 id{blockIdx.x * blockDim.x + threadIdx.x,
                   blockIdx.y * blockDim.y + threadIdx.y};

    if (id.x >= detectorSize.x ||
        id.y >= detectorSize.y)
        return;

    //Add 0.5 to center the pixels
    const float3 pixelPosition = detectorOrigin + pixelDirectionU * (uint2float(id.x) + 0.5f) + pixelDirectionV * (uint2float(id.y) + 0.5f);
    const float3 direction = normalize(pixelPosition - sourcePosition);

    float dir_dot = fabsf(dot(direction, normalize(cross(pixelDirectionU, pixelDirectionV))));

    target[id.x + id.y * detectorSize.x] = source[id.x + id.y * detectorSize.x] / dir_dot;
}
}

void apply_cosweighting(const float3 sourcePosition,
                        const float3 detectorOrigin,
                        const float3 pixelDirectionU,
                        const float3 pixelDirectionV,
                        const int2 detectorSize,
                        float* source,
                        float* target) {
    dim3 dimBlock(16, 16);
    dim3 dimGrid(static_cast<unsigned int>(1 + (detectorSize.x / dimBlock.x)),
                 static_cast<unsigned int>(1 + (detectorSize.y / dimBlock.y)));

    cuda::cosWeightingKernel<<<dimGrid, dimBlock>>>(sourcePosition,
                                                    detectorOrigin,
                                                    pixelDirectionU,
                                                    pixelDirectionV,
                                                    detectorSize,
                                                    source,
                                                    target);
    CUDA_CHECK_ERRORS;
    gpuErrchk(cudaDeviceSynchronize());
}

} //gpumci
