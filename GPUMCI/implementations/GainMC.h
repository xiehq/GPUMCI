#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>
#include <stdint.h>
#include <GPUMCI/physics/CudaSettings.h>
#include <GPUMCI/physics/MaterialEntry.h>
#include <GPUMCI/interactions/PrecomputedInteractionUtils.h>

namespace gpumci {
// forward declaration of the cuda data container
namespace cuda {
struct GainMCCuData;
}

/*
 * GainMC
 *
 * A monte-carlo class that uses precomputed tables to simulate interactions and a fluence map specifiy the number of photons
 */
class GainMC {
  public:
    /*
     * All paramters are given in mm
     *
     * Parameters:
     *   volumeSize     size of the volume (in voxels)
     *   volumeOrigin   position of the lowermost corner of the volume
     *   voxelSize      Size of a voxel
     *   detectorSize   Size of detector (in pixels)
     *   materials      Material attenuation data
     *   rayleighTables Precomputed rayleigh interaction angles
     *   comptonTables  Precomputed compton interaction angles
     *
     */
    GainMC(const Eigen::Vector3i& volumeSize,
           const Eigen::Vector3d& volumeOrigin,
           const Eigen::Vector3d& voxelSize,
           const Eigen::Vector2i& detectorSize,
           const MaterialData& materials,
           const InteractionTables& rayleighTables,
           const InteractionTables& comptonTables);

    /*
     * Update the volume data used by the forward projector
     *
     * Volumes should be given in standard cuda ordering (Fortran/Column major)
     *
     * All data should be given as CUDA pointers
     *
     */
    void setData(const float* densityDevice,
                 const uint8_t* materialTypeDevice);

    /*
     * Perform a projection with a given geometry
     *
     * Parameters:
     *   sourcePosition   Position of the photon source
     *   detectorOrigin   Position of the lowermost corner of the detector
     *   pixelDirectionU  Vector from pixel (i,j) to pixel (i+1,j)
     *   pixelDirectionV  Vector from pixel (i,j) to pixel (i,j+1)
     *   primary          Pointer to the detector where the primary (non-scattered)
     *                    signal should be scored. Represented in (Fortran/Column major order)
     *   scatter          Pointer to the detector where the scattered
     *                    signal should be scored. Represented in (Fortran/Column major order)
     *   gain             Pointer to storage for the gain image (number of photons that should be simulated)
     *   energyMin, energyMax   minumum and maximum energy used
     */
    void project(const Eigen::Vector3d& sourcePosition,
                 const Eigen::Vector3d& detectorOrigin,
                 const Eigen::Vector3d& pixelDirectionU,
                 const Eigen::Vector3d& pixelDirectionV,
                 float* const primary,
                 float* const scatter,
                 const float* const gain,
                 const float energyMin,
                 const float energyMax) const;

  private:
    cuda::CudaParameters _param;
    const Eigen::Vector2i _detectorSize;
    std::shared_ptr<cuda::GainMCCuData> _cudaData;
};
}
