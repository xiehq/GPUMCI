#include <cuda_runtime.h>
#include <curand_kernel.h>

//Thrust
#include <odl_cpp_utils/cuda/disableThrustWarnings.h>
#include <thrust/device_vector.h>
#include <odl_cpp_utils/cuda/enableThrustWarnings.h>

namespace gpumci {
namespace cuda {
//Initialize the curandState vector
__global__ void
curand_init_Kernel(unsigned seed, unsigned offset, unsigned numThreads, unsigned thread_offset, curandState_t* state) {
    unsigned idx = thread_offset + (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= numThreads)
        return;

    curand_init(seed, idx, offset, &state[idx]);
}

//A non-safe version of the init code. This one is significantly (several orders of magnitude) faster, but does not have
//as good statistical guarantees
// This general template works for curandStateMRG32k3a_t, curandStatePhilox4_32_10_t and curandStateXORWOW_t
__global__ void
curand_init_fast_Kernel(unsigned seed, unsigned offset, unsigned numThreads, unsigned thread_offset, curandState_t* state) {
    unsigned idx = thread_offset + (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= numThreads)
        return;

    curand_init(seed * numThreads + idx, 0, offset, &state[idx]);
}
}
}
