#pragma once

#include <time.h>

#include <random>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <odl_cpp_utils/cuda/errcheck.h>

//Thrust
#include <odl_cpp_utils/cuda/disableThrustWarnings.h>
#include <thrust/device_vector.h>
#include <odl_cpp_utils/cuda/enableThrustWarnings.h>

namespace gpumci {
namespace cuda {
/**
 * A rng that always returns the same value, used as a fill-in
 */
struct DeterministicRng {
    // inits the rng
    __device__ void init(int idx) {
    }

    // returns one random integer [0,UINT_MAX]
    __device__ unsigned rand_int() {
        return 0;
    }

    // returns one random number [0,1]
    __device__ float rand() {
        return 0.5f;
    }

    // saves the state of the random number generator, per thread
    // should be called if the rng is to be used by another kernel call later
    __device__ void saveState(int idx) {
    }
};
}
}
