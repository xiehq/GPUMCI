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
struct DosePhaseSpaceMCCuData;
}

/*
 * DosePhaseSpaceMC
 *
 * A monte-carlo class that uses precomputed tables to simulate interactions and a precomputed phase-space to generate particles.
 */
class DosePhaseSpaceMC {
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
     *   n_threads      Lets you set the number of threads to run in parallel,
     *                  forced to be less than or equal to the number of
     *                  particles in the phsp data
     *   materials      Material attenuation data
     *   rayleighTables Precomputed rayleigh interaction angles
     *   comptonTables  Precomputed compton interaction angles
     *
     */
    DosePhaseSpaceMC(const Eigen::Vector3i& volumeSize,
                     const Eigen::Vector3d& volumeOrigin,
                     const Eigen::Vector3d& voxelSize,
                     const Eigen::Vector2i& detectorSize,
                     unsigned n_runs,
                     unsigned n_threads,
                     const MaterialData& materials,
                     const InteractionTables& rayleighTables,
                     const InteractionTables& comptonTables);

    /*
     * All paramters are given in mm, original constructor number of threads is based on detector size
     *
     * Parameters:
     *   volumeSize     size of the volume (in voxels)
     *   volumeOrigin   position of the lowermost corner of the volume
     *   voxelSize      Size of a voxel
     *   detectorSize   Size of detector (in pixels)
     *   n_runs         Number of runs that should be performed
     *                  (this is the number of times the partice is resused)
     *   materials      Material attenuation data
     *   rayleighTables Precomputed rayleigh interaction angles
     *   comptonTables  Precomputed compton interaction angles
     *
     */
    DosePhaseSpaceMC(const Eigen::Vector3i& volumeSize,
                     const Eigen::Vector3d& volumeOrigin,
                     const Eigen::Vector3d& voxelSize,
                     const Eigen::Vector2i& detectorSize,
                     unsigned n_runs,
                     const MaterialData& materials,
                     const InteractionTables& rayleighTables,
                     const InteractionTables& comptonTables);

    // void rotateParticles(std::vector<cuda::CudaMonteCarloParticle>& particles, float3& axis, float angle);

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
                 const std::vector<cuda::CudaMonteCarloParticle>& particles,
                 float* primary,
                 float* scatter,
                 float* dosevolume) const;

    void project(const Eigen::Vector3d& sourcePosition,
                 const Eigen::Vector3d& detectorOrigin,
                 const Eigen::Vector3d& pixelDirectionU,
                 const Eigen::Vector3d& pixelDirectionV,
                 const std::vector<cuda::CudaMonteCarloParticle>& particles,
                 float* primary,
                 float* scatter,
                 float* secondary,
                 float* dosevolume) const;

  private:
    cuda::CudaParameters _param;
    const Eigen::Vector2i _detectorSize;
    const unsigned _nThreads; // gjb
    const unsigned _nRuns;
    std::shared_ptr<cuda::DosePhaseSpaceMCCuData> _cudaData;
};
} // namespace gpumci
