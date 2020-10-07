#pragma once

#include <odl_cpp_utils/cuda/cutil_math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace gpumci {
namespace cuda {

struct DetectorCBCT {
    __host__ DetectorCBCT(float3 detectorOrigin,
                          float3 detectorVectorY,
                          float3 detectorVectorV,
                          float2 inversePixelSize,
                          int2 detectorSize,
                          unsigned pitch,
                          float* const detector)
        : _detectorOrigin(detectorOrigin),
          _detectorNormal(normalize(cross(detectorVectorY, detectorVectorV))),
          _detectorVectorY(normalize(detectorVectorY)),
          _detectorVectorV(normalize(detectorVectorV)),
          _inversePixelSize(inversePixelSize),
          _detectorSize(detectorSize),
          _pitch(pitch),
          _detector(detector) {
    }

    //  Scores the particle on the detector
    template <typename Particle>
    __device__ void scoreDetector(const Particle& photon) {
        if (photon.weight <= 0.0f || //Irrelvant
            photon.energy == 0.0f)   //To low energy
            return;

        const float dir_dot = dot(photon.direction, _detectorNormal);

        if (dir_dot <= 0 || dir_dot >= 2.0) //Traveling wrong way or some error clearly occurred
            return;

        const float distance = dot(_detectorOrigin - photon.position, _detectorNormal);
        const float s = distance / dir_dot;
        const float3 pos = photon.position + s * photon.direction;

        const int i = __float2int_rd(dot(pos - _detectorOrigin, _detectorVectorY) * _inversePixelSize.x);
        const int j = __float2int_rd(dot(pos - _detectorOrigin, _detectorVectorV) * _inversePixelSize.y);

        //printf("%f %f %f %f %f %f %d %d\n",photon.position.x, photon.position.y, photon.position.z, pos.x, pos.y, pos.z, i,j);

        if (i < 0 || j < 0 || i >= _detectorSize.x || j >= _detectorSize.y)
            return; //Photon out of bounds

        //Calculate response function
        const float data = photon.weight * photon.energy / dir_dot;

        //First assign pointer for better parallelism.
        const unsigned index = j * _pitch + i;

        atomicAdd(&_detector[index], data);
    }

  private:
    const float3 _detectorOrigin;
    const float3 _detectorNormal;
    const float3 _detectorVectorY;
    const float3 _detectorVectorV;
    const float2 _inversePixelSize;
    const int2 _detectorSize;
    const unsigned int _pitch;
    float* const _detector;
};
}
}
