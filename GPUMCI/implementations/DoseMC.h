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
struct DoseMCCuData;
}

/*
 * DoseMC
 *
 * A monte-carlo class that uses precomputed tables to simulate interactions.
 */
class DoseMC {
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
    DoseMC(const Eigen::Vector3i& volumeSize,
           const Eigen::Vector3d& volumeOrigin,
           const Eigen::Vector3d& voxelSize,
           const Eigen::Vector2i& detectorSize,
           int n_runs,
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

    void setData(const std::vector<float>& densityHost, const std::vector<uint8_t>& materialTypeHost);

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
     */
    void project(const Eigen::Vector3d& sourcePosition,
                 const Eigen::Vector3d& detectorOrigin,
                 const Eigen::Vector3d& pixelDirectionU,
                 const Eigen::Vector3d& pixelDirectionV,
                 float* primary,
                 float* scatter,
                 float* dosevolume) const;

  private:
    const cuda::CudaParameters _param;
    const int _nRuns;
    std::shared_ptr<cuda::DoseMCCuData> _cudaData;
};
} // namespace gpumci
