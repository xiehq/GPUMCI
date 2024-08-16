#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <odl_cpp_utils/cuda/cutil_math.h>

namespace gpumci {
namespace cuda {

template <typename Particle>
struct DetectorStorePhotonsDevice {
    /*
     * Parameters:
     *   particles_in   Pointer to data storage for the resulting photons
     *   nthreads       Total number of threads, used as a stride for storing photons
     */
    __host__ DetectorStorePhotonsDevice(Particle* particles_out,
                                        unsigned nthreads)
        : _particles_out(particles_out),
          _nthreads(nthreads),
          _current_offset(0) {
    }

    //  Scores the particle by storing to disk.
    __device__ void scoreDetector(const Particle& photon) {
        printf("Hello from the DetectorStorePhoton!\n");
        const unsigned idx = (blockIdx.x * blockDim.x) + threadIdx.x + _current_offset;
        _particles_out[idx] = photon;

        _current_offset += _nthreads;
    }

  private:
    Particle* _particles_out;
    const unsigned _nthreads;
    unsigned _current_offset;
};

/*
A detector that stores the result of the photon to memory after finishing the simulation.

The results are not automatically written but need to be manually copied back to host.
*/
template <typename Particle>
struct DetectorStorePhotons {
    DetectorStorePhotons(unsigned nparticles,
                         unsigned nthreads)
        : _particles_out_device{nparticles},
          _nthreads(nthreads) {
        assert(nthreads > 0);
        assert(nparticles > 0);
    }

    DetectorStorePhotonsDevice<Particle> deviceSide() {
        return {thrust::raw_pointer_cast(_particles_out_device.data()), _nthreads};
    }

    void copy_to_host(std::vector<Particle>& particles_out) const {
        assert(_particles_out_device.size() == particles_out.size());

        thrust::copy(_particles_out_device.begin(),
                     _particles_out_device.end(),
                     particles_out.begin());
    }

  private:
    thrust::device_vector<Particle> _particles_out_device;
    const unsigned _nthreads;
};
} // namespace cuda
} // namespace gpumci
