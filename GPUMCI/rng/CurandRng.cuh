#pragma once

#include <time.h>

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
 * Rng
 */
struct curandRngDevice {
    __host__ curandRngDevice(curandState_t* states)
        : _states(states) {
    }

    // inits the rng
    __device__ void init(int idx) {
        _state = _states[idx];
    }

    // returns one random integer [0,UINT_MAX]
    __device__ unsigned int rand_int() {
        return curand(&_state);
    }

    // returns one random number [0,1]
    __device__ float rand() {
        return curand_uniform(&_state);
    }

    // saves the state of the random number generator, per thread
    // should be called if the rng is to be used by another kernel call later
    __device__ void saveState(int idx) {
        _states[idx] = _state;
    }

  private:
    curandState_t* _states;
    curandState_t _state;
};

//Init kernel is in CurandRng.cu
__global__ void curand_init_Kernel(unsigned seed, unsigned offset, unsigned numThreads, unsigned thread_offset, curandState_t* state);
__global__ void curand_init_fast_Kernel(unsigned seed, unsigned offset, unsigned numThreads, unsigned thread_offset, curandState_t* state);

/**
 * Rng
 */
struct curandRng {
    /*
     Initializes enough RNG's for numThreads.

     Seed indicates the random seed, setting it to 0 causes a random seed to be chosen.

     Offset indicates if the sequence should be offset.
    */
    curandRng(unsigned numThreads, unsigned seed = 0, unsigned offset = 0)
        : _states(numThreads) {
        static const unsigned num_threads_per_run = 50000;
        if (seed == 0)
            seed = static_cast<unsigned>(time(NULL));

        unsigned n_runs = numThreads / num_threads_per_run + ((numThreads % num_threads_per_run) != 0);

        //Calculate maximum occupancy settings
        int blockSize;   // The launch configurator returned block size
        int minGridSize; // The minimum grid size needed to achieve the
                         // maximum occupancy for a full device launch
        int gridSize;    // The actual grid size needed, based on input size

        CUDA_SAFE_CALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize,
                                                          &blockSize,
                                                          curand_init_fast_Kernel,
                                                          0,
                                                          num_threads_per_run));
        // Round up according to array size
        gridSize = (num_threads_per_run + blockSize - 1) / blockSize;

        for (unsigned run = 0; run < n_runs; ++run) {
            curand_init_fast_Kernel << <gridSize, blockSize>>> (seed,
                                                                offset,
                                                                numThreads,
                                                                num_threads_per_run * run,
                                                                thrust::raw_pointer_cast(_states.data()));
        }
        CUDA_KERNEL_ERRCHECK;
    }

    curandRngDevice deviceSide() {
        return curandRngDevice(thrust::raw_pointer_cast(_states.data()));
    }

  private:
    thrust::device_vector<curandState_t> _states;
};
}
}
