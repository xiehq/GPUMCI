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
struct PhaseSpaceStorePhotonsMCCuData;
}

/*
 * PhaseSpaceMC
 *
 * A monte-carlo class that uses precomputed tables to simulate interactions and a precomputed phase-space to generate particles.
 * The result is stored in an array.
 */
class PhaseSpaceStorePhotonsMC {
  public:
    /*
     * All paramters are given in mm
     *
     * Parameters:
     *   volumeSize     size of the volume (in voxels)
     *   volumeOrigin   position of the lowermost corner of the volume
     *   voxelSize      Size of a voxel
     *   materials      Material attenuation data
     *   rayleighTables Precomputed rayleigh interaction angles
     *   comptonTables  Precomputed compton interaction angles
     *
     */
    PhaseSpaceStorePhotonsMC(const Eigen::Vector3i& volumeSize,
                             const Eigen::Vector3d& volumeOrigin,
                             const Eigen::Vector3d& voxelSize,
                             const MaterialData& materials,
                             const InteractionTables& rayleighTables,
                             const InteractionTables& comptonTables);

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
     *   particles_in     Description of the incoming particles
     *   particles_out    Storage for the outgoing particles
     */
    void project(const std::vector<cuda::CudaMonteCarloParticle>& particles_in,
                 std::vector<cuda::CudaMonteCarloParticle>& particles_out) const;

  private:
    cuda::CudaParameters _param;
    std::shared_ptr<cuda::PhaseSpaceStorePhotonsMCCuData> _cudaData;
};
} // namespace gpumci
