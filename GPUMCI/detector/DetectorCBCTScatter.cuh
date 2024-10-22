#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <odl_cpp_utils/cuda/cutil_math.h>
#include <stdio.h>

namespace gpumci {
namespace cuda {

struct DetectorCBCTScatter {
    __host__ DetectorCBCTScatter(float3 detectorOrigin,
                                 float3 detectorVectorU,
                                 float3 detectorVectorV,
                                 float2 inversePixelSize,
                                 int2 detectorSize,
                                 unsigned pitch,
                                 float* const primaryDetector,
                                 float* const secondaryDetector)
        : _detectorOrigin(detectorOrigin),
          _detectorNormal(normalize(cross(detectorVectorU, detectorVectorV))),
          _detectorVectorU(normalize(detectorVectorU)),
          _detectorVectorV(normalize(detectorVectorV)),
          _inversePixelSize(inversePixelSize),
          _detectorSize(detectorSize),
          _pitch(pitch),
          _primaryResult(primaryDetector),
          _secondaryResult(secondaryDetector),
          _tertiaryResult(nullptr) { // Default value
    }

    // Overloaded constructor with an additional parameter for secondary photons
    __host__ DetectorCBCTScatter(float3 detectorOrigin,
                                 float3 detectorVectorU,
                                 float3 detectorVectorV,
                                 float2 inversePixelSize,
                                 int2 detectorSize,
                                 unsigned pitch,
                                 float* const primaryDetector,
                                 float* const secondaryDetector,
                                 float* const tertiaryDetector) // New parameter
        : _detectorOrigin(detectorOrigin),
          _detectorNormal(normalize(cross(detectorVectorU, detectorVectorV))),
          _detectorVectorU(normalize(detectorVectorU)),
          _detectorVectorV(normalize(detectorVectorV)),
          _inversePixelSize(inversePixelSize),
          _detectorSize(detectorSize),
          _pitch(pitch),
          _primaryResult(primaryDetector),
          _secondaryResult(secondaryDetector),
          _tertiaryResult(tertiaryDetector) { // New member
    }

    //  Scores the particle on the detector
    template <typename Particle>
    __device__ void scoreDetector(const Particle& myPhoton) {
        if (myPhoton.weight <= 0.0f || // Irrelvant
            myPhoton.energy == 0.0f)   // To low energy
            return;

        float3 direction = normalize(myPhoton.direction);

        const float dir_dot = dot(direction, _detectorNormal);

        if (dir_dot <= 0.0 || dir_dot >= 2.0) // Traveling wrong way or some error clearly occurred
            return;

        const float distance = dot(_detectorOrigin - myPhoton.position, _detectorNormal);
        const float s = distance / dir_dot;
        const float3 pos = myPhoton.position + s * direction;

        if (!isfinite(pos.x) || !isfinite(pos.y) || !isfinite(pos.z))
            return;

        const int i = __float2int_rd(dot(pos - _detectorOrigin, _detectorVectorU) * _inversePixelSize.x);
        const int j = __float2int_rd(dot(pos - _detectorOrigin, _detectorVectorV) * _inversePixelSize.y);
        if (i < 0 || j < 0 || i >= _detectorSize.x || j >= _detectorSize.y) return; // Photon out of bounds

        // Calculate response function
        const float E = myPhoton.energy;
        const float energy_response = E; //* (1.0f - expf(-(E / 0.035f) * (E / 0.035f))) * expf(-E / 0.053f); //f0.03f + myPhoton.energy;
        const float data = myPhoton.weight * energy_response / dir_dot;

        // First assign pointer for better parallelism.
        const unsigned index = j * _pitch + i;
        /* If the boolean expression "myPhoton.primary" is true, the pointer "pointer" will be set
        to the address of the element at index "index" of the "_primaryResult" array. If the expression
        is false, "pointer" will be set to the address of the element at index "index" of the "_secondaryResult" array.
        if (myPhoton.isTertiary(myPhoton.data)) {
            printf("Yes Tertiary.\n");
        } else {
            printf("Not Tertiary.\n");
        }*/
        float* pointer;
        if (myPhoton.isTertiary()) {
            pointer = &_tertiaryResult[index];
        } else if (myPhoton.primary) {
            pointer = &_primaryResult[index];
        } else {
            pointer = &_secondaryResult[index];
        }

        atomicAdd(pointer, data);
    }

  private:
    const float3 _detectorOrigin;
    const float3 _detectorNormal;
    const float3 _detectorVectorU;
    const float3 _detectorVectorV;
    const float2 _inversePixelSize;
    const int2 _detectorSize;
    const unsigned int _pitch;
    float* const _primaryResult;
    float* const _secondaryResult;
    float* const _tertiaryResult; // New member
};
} // namespace cuda
} // namespace gpumci
