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
namespace {
std::vector<unsigned> get_primes(unsigned limit = std::numeric_limits<unsigned>::max()) {
    std::vector<bool> is_prime(limit, true);
    std::vector<unsigned> primes;

    const unsigned sqrt_limit = static_cast<unsigned>(std::sqrt(limit));
    for (unsigned n = 2; n <= sqrt_limit; ++n)
        if (is_prime[n]) {
            primes.push_back(n);

            for (unsigned k = n * n, ulim = limit; k < ulim; k += n)
                //NOTE: "unsigned" is used to avoid an overflow in `k+=n` for `limit` near INT_MAX
                is_prime[k] = false;
        }

    for (unsigned n = sqrt_limit + 1; n < limit; ++n)
        if (is_prime[n])
            primes.push_back(n);

    return primes;
}
}
/**
 * Rng
 */
struct MWCRngDevice {
    __host__ MWCRngDevice(uint2 *mults, uint2 *seeds)
        : _mults(mults),
          _seeds(seeds) {
    }

    // inits the rng
    __device__ void init(int idx) {
        _a = _mults[idx];
        _z = _seeds[idx];
    }

    // returns one random integer [0,UINT_MAX]
    __device__ unsigned rand_int() {
        _z.x = (_a.x * (_z.x & 65535)) + (_z.x >> 16);
        _z.y = (_a.y * (_z.y & 65535)) + (_z.y >> 16);
        return ((_z.y << 16) + _z.x);
    }

    // returns one random number [0,1]
    __device__ float rand() {
        return rand_int() * 2.328306e-10f; // 1 / 2^32
    }

    // saves the state of the random number generator, per thread
    // should be called if the rng is to be used by another kernel call later
    __device__ void saveState(int idx) {
        _seeds[idx] = _z;
    }

  private:
    uint2* const _mults;
    uint2* const _seeds;
    uint2 _z, _a;
};

/**
 * Rng
 */
struct MWCRng {
    /*
    Initializes enough RNG's for numThreads.

    Seed indicates the random seed, setting it to 0 causes a random seed to be chosen.
    */
    MWCRng(unsigned numThreads, unsigned seed = 0) {
        if (seed == 0)
            seed = static_cast<unsigned>(time(NULL));

        std::vector<unsigned> safe_primes = get_primes(1 << 16);
        std::vector<uint2> hMults;
        std::vector<uint2> hSeeds;

        auto host_rng = std::minstd_rand{seed};

        unsigned int numPrimes = static_cast<unsigned int>(safe_primes.size());

        for (size_t i = 0; i < numThreads; i++) {
            hMults.push_back(uint2{safe_primes[host_rng() % numPrimes],
                                   safe_primes[host_rng() % numPrimes]});
            hSeeds.push_back(uint2{(unsigned)host_rng(), (unsigned)host_rng()});
        }

        _mults.resize(numThreads);
        _seeds.reserve(numThreads);

        thrust::copy(hMults.begin(), hMults.end(), _mults.begin());
        thrust::copy(hSeeds.begin(), hSeeds.end(), _seeds.begin());
    }

    MWCRngDevice deviceSide() {
        return MWCRngDevice(thrust::raw_pointer_cast(_mults.data()),
                            thrust::raw_pointer_cast(_seeds.data()));
    }

  private:
    thrust::device_vector<uint2> _mults;
    thrust::device_vector<uint2> _seeds;
};
}
}
