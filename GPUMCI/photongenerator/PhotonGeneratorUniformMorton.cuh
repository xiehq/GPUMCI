#pragma once

#include <cstdint>
#include <odl_cpp_utils/cuda/cutil_math.h>
#include <GPUMCI/utils/cuda_utils.cuh>

#include <GPUMCI/physics/CudaSettings.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace gpumci {
namespace cuda {
namespace {
__device__ uint32_t CompactBy1(uint32_t x) {
    x &= 0x55555555;                 // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x = (x ^ (x >> 1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x >> 2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x >> 4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x >> 8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
    return x;
}
__device__ void MortonDecode2(const uint32_t c, uint32_t& x, uint32_t& y) {
    x = CompactBy1(c);
    y = CompactBy1(c >> 1);
}
}

struct PhotonGeneratorUniformMortonSharedData {
};

/*
 * Generates an uniform number of photons per pixel, this version uses a Morton curve to optimzie data locality.
 *
 * For this to work, number of threads had to, in each dimension, be a power of two larger or equal to detectorSize
 *
 */
struct PhotonGeneratorUniformMorton {
    typedef PhotonGeneratorUniformMortonSharedData SharedData;
    typedef CudaMonteCarloScatterParticle Particle;

    __host__ PhotonGeneratorUniformMorton(int2 detectorSize,
                                          float3 detectorOrigin,
                                          float3 detectorVectorU,
                                          float3 detectorVectorV,
                                          float3 sourcePosition,
                                          int nRunsPerThread)
        : _detectorSize(detectorSize),
          _detectorOrigin(detectorOrigin),
          _detectorVectorU(detectorVectorU),
          _detectorVectorV(detectorVectorV),
          _sourcePosition(sourcePosition),
          _nRunsPerThread(nRunsPerThread) {
    }

    __device__ void init(int idx, SharedData& sharedData, const CudaParameters& c_param) {
        uint32_t row, col;
        MortonDecode2(idx, row, col);
        if (row < _detectorSize.x && col < _detectorSize.y) {
            _ry = (float)row;
            _rz = (float)col;

            _numLeft = _nRunsPerThread;
        } else
            _numLeft = 0;
    }

    template <typename Rng>
    __device__ bool generatePhoton(Particle& photon, int idx, Rng& rng, SharedData& sharedData, const CudaParameters& c_param) {
        if (--_numLeft < 0)
            return false;

        //When using even sampling the photon is selected at a one pixel per thread basis.
        photon.weight = 1.0f;

        const float3 detectorpos = _detectorOrigin +
                                   (_ry + rng.rand()) * _detectorVectorU +
                                   (_rz + rng.rand()) * _detectorVectorV;
        const float3 start = _sourcePosition;

        photon.direction = normalize(detectorpos - start);

        const bool hit = findVolumeIntersection(start, photon.direction, c_param.volumeMin, c_param.volumeMax, photon.position);

        //If the photon misses the volume, transport it to the detector and score without any extra effort.
        if (!hit) {
            photon.position = detectorpos - photon.direction * 0.01; //Particle is moved just in front of the detector.
        }

        //triangular distribution (asssumes no bowtie)

        //        printf("%f %f %f %f %f %f %f %f\n", photon.energy, photon.weight, photon.position.x, photon.position.y, photon.position.z, photon.direction.x, photon.direction.y, photon.direction.z);
        photon.energy = 0.08 + 0.03f * (1.0f - sqrtf(1.0f - rng.rand()));
        photon.primary = true;

        return true;
    }

  private:
    const int2 _detectorSize;
    const float3 _detectorOrigin;
    const float3 _detectorVectorU;
    const float3 _detectorVectorV;
    const float3 _sourcePosition;
    const unsigned _nRunsPerThread;
    float _ry, _rz;
    int _numLeft;
};
}
}
