#pragma once

#include <odl_cpp_utils/cuda/cutil_math.h>

#include <GPUMCI/physics/CudaSettings.h>
#include <GPUMCI/utils/cuda_utils.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace gpumci {
namespace cuda {
struct PhotonGeneratorUniformSharedData {
};

struct PhotonGeneratorUniform {
    typedef PhotonGeneratorUniformSharedData SharedData;
    typedef CudaMonteCarloScatterParticle Particle;

    __host__ PhotonGeneratorUniform(int2 detectorSize,
                                    float3 detectorOrigin,
                                    float3 detectorVectorU,
                                    float3 detectorVectorV,
                                    float3 sourcePosition,
                                    int nRunsPerThread,
                                    float energyMin = 0.08,
                                    float energySpread = 0.03)
        : _detectorSize(detectorSize),
          _detectorOrigin(detectorOrigin),
          _detectorVectorU(detectorVectorU),
          _detectorVectorV(detectorVectorV),
          _sourcePosition(sourcePosition),
          _nRunsPerThread(nRunsPerThread),
          _energyMin(energyMin),
          _energySpread(energySpread) {
    }

    __device__ void init(int idx, SharedData& sharedData, const CudaParameters& c_param) {
        idx %= (_detectorSize.x * _detectorSize.y); //In case we want multiple threads per pixel

        _ry = (float)(idx % _detectorSize.x);
        _rz = (float)(idx / _detectorSize.x);
        _numLeft = _nRunsPerThread;
    }

    template <typename Rng>
    __device__ bool generatePhoton(Particle& photon, int idx, Rng& rng, SharedData& sharedData, const CudaParameters& c_param) {
        if (--_numLeft < 0)
            return false;

        //When using even sampling the photon is selected at a one pixel per thread basis.
        photon.weight = 1.0f;

        const float3 detectorpos = _detectorOrigin +
                                   (_ry + 0.5f) * _detectorVectorU +
                                   (_rz + 0.5f) * _detectorVectorV;
        const float3 start = _sourcePosition;

        photon.direction = normalize(detectorpos - start);

        const bool hit = util::findVolumeIntersection(start, photon.direction, c_param.volumeMin, c_param.volumeMax, photon.position);

        //If the photon misses the volume, transport it to the detector and score without any extra effort.
        if (!hit) {
            photon.position = detectorpos - photon.direction * 0.01; //Particle is moved just in front of the detector.
        }

        //triangular distribution (asssumes no bowtie)
        photon.energy = _energyMin + _energySpread * (1.0f - sqrtf(1.00001f - rng.rand()));
        photon.primary = true;

        //printf("%f %f %f %f %f %f %f %f\n", photon.energy, photon.weight, photon.position.x, photon.position.y, photon.position.z, photon.direction.x, photon.direction.y, photon.direction.z);
        return true;
    }

  private:
    const int2 _detectorSize;
    const float3 _detectorOrigin;
    const float3 _detectorVectorU;
    const float3 _detectorVectorV;
    const float3 _sourcePosition;
    const unsigned _nRunsPerThread;
    const float _energyMin, _energySpread;
    float _ry, _rz;
    int _numLeft;
};
}
}
