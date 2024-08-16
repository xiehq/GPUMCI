#pragma once

#include <cuda_runtime.h>

namespace gpumci {
namespace cuda {

struct NoDetector {
    //  Scores the particle on the detector
    template <typename Particle>
    __device__ void scoreDetector(Particle& photon) {
        // No-op
        printf("Hello from the NoDetector!\n");
    }
};
} // namespace cuda
} // namespace gpumci
