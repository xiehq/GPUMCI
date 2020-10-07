#pragma once

#include <odl_cpp_utils/cuda/cutil_math.h>

#include <GPUMCI/physics/CudaSettings.h>
#include <GPUMCI/utils/cuda_utils.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace gpumci {
namespace cuda {

struct PhotonGeneratorGainImagesharedData {
};

//
// Analytic photon generator that generates photons with distribution according to a gain image. The energy is picked from a triangular distribution (hat function), with min energyMin and max energyMin+energySpread
//
// To simulate a whole spectrum, several of these can be used togeather.
//
// The number of photons is equal to the values of the gain image, weights are scaled to compensate for non integer values
//
struct PhotonGeneratorGainImage {
    typedef PhotonGeneratorGainImagesharedData SharedData;
    typedef CudaMonteCarloScatterParticle Particle;

    __host__ PhotonGeneratorGainImage(int2 detectorSize,
                                      float3 detectorOrigin,
                                      float3 detectorVectorU,
                                      float3 detectorVectorV,
                                      float3 sourcePosition,
                                      const float* const gain_image,
                                      float energyMin,
                                      float energySpread)
        : _detectorSize(detectorSize),
          _detectorOrigin(detectorOrigin),
          _detectorVectorU(detectorVectorU),
          _detectorVectorV(detectorVectorV),
          _sourcePosition(sourcePosition),
          _gain_image(gain_image),
          _energyMin(energyMin),
          _energySpread(energySpread) {
    }

    __device__ void init(int idx,
                         SharedData& sharedData,
                         const CudaParameters& c_param) {
        _ry = (float)(idx % _detectorSize.x);
        _rz = (float)(idx / _detectorSize.x);

        //When using even gain image sampling the weight is set by the gain image
        float gain = _gain_image[idx];

        //Round value up to ensure we simulate at least one particle
        _numLeft = __float2int_ru(gain);

        //Scale weights for remaining values
        _weight = gain / static_cast<float>(_numLeft);
    }

    template <typename Rng>
    __device__ bool generatePhoton(Particle& photon, int idx, Rng& rng, SharedData& sharedData, const CudaParameters& c_param) {
        if (--_numLeft < 0)
            return false;

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

        photon.energy = _energyMin + _energySpread * (1.0f - sqrtf(1.0f - rng.rand()));
        photon.weight = _weight;
        photon.primary = true;

        return true;
    }

  private:
    const int2 _detectorSize;
    const float3 _detectorOrigin;
    const float3 _detectorVectorU;
    const float3 _detectorVectorV;
    const float3 _sourcePosition;
    const float* const _gain_image;
    const float _energyMin, _energySpread;

    float _ry, _rz;
    int _numLeft;
    float _weight;
};
}
}
