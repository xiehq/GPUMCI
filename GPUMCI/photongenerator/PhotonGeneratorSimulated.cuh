#pragma once

#include <GPUMCI/physics/CudaSettings.h>
#include <GPUMCI/utils/cuda_utils.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace gpumci {
namespace cuda {

struct SimulatedPhotonGeneratorSharedData {
    int num_photons_left;
};

struct PhotonGeneratorSimulated {
    typedef SimulatedPhotonGeneratorSharedData SharedData;
    typedef CudaMonteCarloScatterParticle Particle;

    __host__ PhotonGeneratorSimulated(int numPhotons)
        : _numPhotons(numPhotons) {
    }

    __device__ void init(int idx, SharedData& sharedData, const CudaParameters& parms) {
        _currentId = idx;

        if (threadIdx.x == 0)
            sharedData.num_photons_left = blockDim.x * parms.nRunsPerThread;

        __syncthreads();
    }

    template <typename Rng>
    __device__ bool generatePhoton(Particle& photon, int idx, Rng& rng, SharedData& sharedData, const CudaParameters& c_param) {
        if (atomicSub(&sharedData.num_photons_left, 1) <= 0)
            return false;

        //When using even sampling the photon is selected at a one pixel per thread basis.
        const float ry = ((float)(_currentId % c_param.detectorSize.x) + rng.rand());
        const float rz = ((float)(_currentId / c_param.detectorSize.x) + rng.rand());
        photon.weight = 1.0f;

        const float3 detectorpos = c_param.detectorOrigin +
                                   ry * c_param.detectorVectorU +
                                   rz * c_param.detectorVectorV;
        const float3 start = c_param.sourcePosition;

        photon.direction = normalize(detectorpos - start);

        const bool hit = util::findVolumeIntersection(start, photon.direction, c_param.volumeMin, c_param.volumeMax, photon.position);

        //If the photon misses the volume, transport it to the detector and score without any extra effort.
        if (!hit) {
            photon.position = detectorpos - photon.direction * 0.01; //Particle is moved just in front of the detector.
        }

        //triangular distribution (asssumes no bowtie)
        photon.energy = 0.08 + 0.03f * (1.0f - sqrtf(1.0f - rng.rand()));
        photon.primary = true;

        //Avoid alliasing effects. 877 is relatively prime to detector size and smaller than numPhotons.
        _currentId = (_currentId * 877) % _numPhotons;

        return true;
    }

  private:
    int _currentId;
    const int _numPhotons;
};
}
}
