#pragma once

#include <odl_cpp_utils/cuda/cutil_math.h>

#include <GPUMCI/physics/CudaSettings.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace gpumci {
namespace cuda {

/*
 Add a dose detector to a interaction handler
*/
template <typename InteractionHandler>
struct DoseDetector {
    __host__ DoseDetector(const float3 volumeMin,
                          const float3 inverseVoxelSize,
                          const int3 volumeSize,
                          float* const dose_volume,
                          const InteractionHandler& interactionHandler)
        : _volumeMin(volumeMin),
          _inverseVoxelSize(inverseVoxelSize),
          _volumeSize(volumeSize),
          _dose_volume(dose_volume),
          _interactionHandler(interactionHandler) {
    }

    template <typename Rng, typename Particle>
    __device__ void simulateInteraction(const float meanFreePathCM,
                                        const float stepCM,
                                        const uint8_t medium,
                                        const float density,
                                        Particle& photon,
                                        Rng& rng,
                                        const CudaParameters& c_param) const {
        float energy_0 = photon.energy;

        _interactionHandler.simulateInteraction(meanFreePathCM,
                                                stepCM,
                                                medium,
                                                density,
                                                photon,
                                                rng,
                                                c_param);

        float energy_diff = energy_0 - photon.energy;

        // printf("Volume Received Dose: %f\n", photon.weight * energy_diff);

        if (energy_diff > 0.0f) {
            int3 ivoxel = getIndicesVoxel(photon.position);
            if (ivoxel.x >= _volumeSize.x ||
                ivoxel.y >= _volumeSize.y ||
                ivoxel.z >= _volumeSize.z)
                return;

            unsigned index = ivoxel.x + ivoxel.y * _volumeSize.x + ivoxel.z * _volumeSize.x * _volumeSize.y;

            // size_t allocated_size = _volumeSize.x * _volumeSize.y * _volumeSize.z;

            // if (index < allocated_size) {
            //     printf("Index %u within bounds (allocated size: %lu)\n", index, allocated_size);
            // } else {
            //     printf("Error: Index %u out of bounds (allocated size: %lu)\n", index, allocated_size);
            // }

            // Print the index and the dose amount being added - for debugging
            // printf("Before Adding: Index: %u, Current Dose: %f\n", index, _dose_volume[index]);
            // printf("Adding dose: %f to index: %u\n", photon.weight * energy_diff, index);
            atomicAdd(&_dose_volume[index], photon.weight * energy_diff);
            // printf("After Adding: Index: %u, New Dose: %f\n", index, _dose_volume[index]);
            // printf("Done Adding:\n");
        }

        // if (energy_diff > 0.0f) {
        //     int3 ivoxel = getIndicesVoxel(photon.position);
        //     if (ivoxel.x >= _volumeSize.x ||
        //         ivoxel.y >= _volumeSize.y ||
        //         ivoxel.z >= _volumeSize.z)
        //         return;

        //     unsigned index = ivoxel.x + ivoxel.y * _volumeSize.x + ivoxel.z * _volumeSize.x * _volumeSize.y;

        //     printf("Before Adding: Index: %u, Current Dose: %f\n", index, _dose_volume[index]);
        //     printf("Adding dose: %f to index: %u\n", photon.weight * energy_diff, index);

        //     atomicAdd(&_dose_volume[index], photon.weight * energy_diff);

        //     printf("After Adding: Index: %u, New Dose: %f\n", index, _dose_volume[index]);
        //     printf("Done Adding!\n");
        // }
    }

  private:
    __device__ int3 getIndicesVoxel(const float3& pos) const {
        return {__float2int_rd((pos.x - _volumeMin.x) * _inverseVoxelSize.x),
                __float2int_rd((pos.y - _volumeMin.y) * _inverseVoxelSize.y),
                __float2int_rd((pos.z - _volumeMin.z) * _inverseVoxelSize.z)};
    }

    const int3 _volumeSize;
    const float3 _volumeMin;
    const float3 _inverseVoxelSize;
    float* const _dose_volume;
    const InteractionHandler _interactionHandler;
};
} // namespace cuda
} // namespace gpumci
