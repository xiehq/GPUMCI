#pragma once

#include <Eigen/Core>
#include <GPUMCI/interactions/PrecomputedInteractionUtils.h>
#include <GPUMCI/physics/CudaSettings.h>
#include <GPUMCI/physics/MaterialEntry.h>
#include <memory>
#include <stdint.h>
#include <vector>

namespace gpumci {
namespace cuda {
struct GainDoseMCCuData;
}

/*
 * GainDoseMC
 *
 * A monte-carlo class that uses precomputed tables to simulate interactions.
 */
class GainDoseMC {
  public:
    /*
     * All paramters are given in mm
     *
     * Parameters:
     *   volumeSize     size of the volume (in voxels)
     *   volumeOrigin   position of the lowermost corner of the volume
     *   voxelSize      Size of a voxel
     *   detectorSize   Size of detector (in pixels)
     *   n_runs         Number of runs that should be performed
     *   materials      Material attenuation data
     *   rayleighTables Precomputed rayleigh interaction angles
     *   comptonTables  Precomputed compton interaction angles
     *
     */
    GainDoseMC(const Eigen::Vector3i& volumeSize,
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
     *   dosevolume       Pointer to the volume where the dose should be written
     *   gain             Pointer to storage for the gain image (number of photons that should be simulated)
     *   energyMin, energyMax   minumum and maximum energy used
     */
    void project(const Eigen::Vector3d& sourcePosition,
                 const Eigen::Vector3d& detectorOrigin,
                 const Eigen::Vector3d& pixelDirectionU,
                 const Eigen::Vector3d& pixelDirectionV,
                 float* dosevolume,
                 const float* const gain,
                 const float energyMin,
                 const float energyMax) const;

  private:
    const cuda::CudaParameters _param;
    const Eigen::Vector2i _detectorSize;
    std::shared_ptr<cuda::GainDoseMCCuData> _cudaData;
};
} // namespace gpumci
