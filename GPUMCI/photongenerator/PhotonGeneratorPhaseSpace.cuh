#pragma once

#define BLOCKWIDTH 256

#include <odl_cpp_utils/cuda/disableThrustWarnings.h>
#include <thrust/device_vector.h>
#include <odl_cpp_utils/cuda/enableThrustWarnings.h>

#include <odl_cpp_utils/utils/cast.h>
#include <GPUMCI/physics/CudaSettings.h>
#include <GPUMCI/utils/cuda_utils.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>

namespace gpumci {
namespace cuda {
struct PhotonGeneratorPhaseSpaceSharedData {
};

struct PhotonGeneratorPhaseSpaceDevice {
    typedef PhotonGeneratorPhaseSpaceSharedData SharedData;
    typedef CudaMonteCarloScatterParticle Particle;

    __host__ PhotonGeneratorPhaseSpaceDevice(const CudaMonteCarloParticle* const phaseSpace,
                                             unsigned runs_per_photon,
                                             unsigned nthreads,
                                             unsigned nphotons)
        : _photons(phaseSpace),
          _runs_per_photon(runs_per_photon),
          _nthreads(nthreads),
          _nphotons(nphotons) {
    }

    __device__ void next_photon(const CudaParameters& c_param) {
        particle = _photons[_current_idx];
        util::findVolumeIntersection(particle.position, particle.direction, c_param.volumeMin, c_param.volumeMax, particle.position);

        _num_photons_left = _runs_per_photon;
    }
    __device__ void init(int idx, SharedData& sharedData, const CudaParameters& c_param) {
        _current_idx = idx;
        next_photon(c_param);
    }

    template <typename Rng>
    __device__ bool generatePhoton(Particle& photon, int idx, Rng& rng, SharedData& shared, const CudaParameters& c_param) {
        if (_num_photons_left == 0) {
            _current_idx += _nthreads;

            if (_current_idx >= _nphotons)
                return false;
            else
                next_photon(c_param);
        }
        _num_photons_left -= 1;

        // Particles are stored without the primary attribute in memory, this is set to true by default
        photon = particle;

        return true;
    }

  private:
    const CudaMonteCarloParticle* const _photons;
    const unsigned _runs_per_photon;
    const unsigned _nthreads;
    const unsigned _nphotons;
    CudaMonteCarloParticle particle;
    unsigned _num_photons_left;
    unsigned _current_idx;
};

struct PhotonGeneratorPhaseSpace {
    PhotonGeneratorPhaseSpace(const std::vector<CudaMonteCarloParticle>& particles,
                              unsigned runs_per_photon,
                              unsigned nthreads)
        : _particles{particles.begin(), particles.end()},
          _runs_per_photon{runs_per_photon},
          _nthreads(nthreads) {
        assert(runs_per_photon > 0);
        assert(nthreads > 0);
    }

    PhotonGeneratorPhaseSpaceDevice deviceSide() {
        return {thrust::raw_pointer_cast(_particles.data()), _runs_per_photon, _nthreads, narrow_cast<unsigned>(_particles.size())};
    }

  private:
    const thrust::device_vector<CudaMonteCarloParticle> _particles;
    const unsigned _runs_per_photon;
    const unsigned _nthreads;
};
}
}
