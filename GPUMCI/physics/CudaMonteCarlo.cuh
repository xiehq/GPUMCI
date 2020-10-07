#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <texture_fetch_functions.h>
#include <device_functions.h>

#include <iostream>

#include <GPUMCI/physics/CudaSettings.h>
#include <odl_cpp_utils/cuda/cutil_math.h>
#include <odl_cpp_utils/cuda/errcheck.h>
#include <cuda_profiler_api.h>

namespace gpumci {
namespace cuda {
namespace {
/**
* Device Functions
*/

//  Polls the geometry to find out in which voxel (indices) the particle at position 'pos' is.
//  Results returned through the parameters
__device__ void getIndicesVoxel(const float3& pos, const float3& volumeMin, const float3& inverseVoxelSize, float3& ivoxel) {
    ivoxel = (pos - volumeMin) * inverseVoxelSize;
}

//  polls the geometry to find out if the particle has left the tracking geometry
__device__ bool particleOutOfBounds(const float3& position, const float3& volumeMin, const float3& volumeMax) {
    return (position.x < volumeMin.x || volumeMax.x < position.x ||
            position.y < volumeMin.y || volumeMax.y < position.y ||
            position.z < volumeMin.z || volumeMax.z < position.z ||
            !isfinite(position.x) || !isfinite(position.y) || !isfinite(position.z));
}

//  Simulates the track of one photon through a geometry.
template <typename PhotonGenerator, typename InteractionHandler, typename Scorer, typename StepLength, typename Rng>
__global__ void
monteCarloKernel(const cudaTextureObject_t texDensityVolume,
                 const cudaTextureObject_t texMaterialVolume,
                 const CudaParameters c_param,
                 const unsigned numThreads,
                 const InteractionHandler interaction,
                 PhotonGenerator photonGenerator,
                 Scorer scorer,
                 StepLength stepLength,
                 Rng rng) {
    typedef typename PhotonGenerator::Particle Particle;

    // Look up the thread index and make sure that the thread is actively simulating a particle:
    const unsigned idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= numThreads)
        return;

    //Initialize the photon generator
    __shared__ typename PhotonGenerator::SharedData sharedData;
    photonGenerator.init(idx, sharedData, c_param);

    //Initialize the RNG
    rng.init(idx);

    //Pre define some variables for efficiency and readability
    float3 ivoxel;
    uint8_t myMedium;
    float myDensity;

    //Empty particle
    Particle myPhoton;

    //[Main Loop]
    while (photonGenerator.generatePhoton(myPhoton, idx, rng, sharedData, c_param)) { //Loop while we still have particles to generate
        int step = 0;

        while (myPhoton.energy >= 0.001 && ++step < 5000) { //c_param.photon_energy_cutoff && ++step < 50) {
            // Look up the Woodcock mean free path for the current photon energy in the current local geometry.
            float meanFreePathCM, stepCM;
            stepLength(myPhoton, meanFreePathCM, stepCM, rng);

            //Advance the particle
            myPhoton.position += (stepCM * 10.0f) * myPhoton.direction;

            //Verify that it is still within the geometric bounds.
            if (particleOutOfBounds(myPhoton.position, c_param.volumeMin, c_param.volumeMax))
                break;

            // Get position in index of photon
            getIndicesVoxel(myPhoton.position, c_param.volumeMin, c_param.inverseVoxelSize, ivoxel);

            // Gather information concerning the current voxel (medium index, mass density, cross sections).
            myDensity = tex3D<float>(texDensityVolume, ivoxel.x, ivoxel.y, ivoxel.z);
            myMedium = tex3D<uint8_t>(texMaterialVolume, ivoxel.x, ivoxel.y, ivoxel.z);

            // Simulate interaction
            interaction.simulateInteraction(meanFreePathCM, stepCM, myMedium, myDensity, myPhoton, rng, c_param);
        }

        // Score the particle
        scorer.scoreDetector(myPhoton);
    } // End : [Main Loop]

    // Save the state of the random number generator.
    rng.saveState(idx);
}

} // runmc_private

/**
 * RunMC, runs the montecarlo engine with given parameters
 *
 * Parameters
 *   densityVolume          3D float texture object with clamp
 *                          Should be of size c_param.volumeSize
 *   MaterialTypeVolume     3D uint8_t texture object
 *                          Should be of size c_param.volumeSize
 *   c_param                Simulation parameters
 *   numberOfThreads        The number of threads that should be used
 *   photonGen              Photon generator
 *   scorer                 Detector object 
 *   stepLength             Object for selecting the step length
 *   rng                    Random number generator
 *
 */
template <typename InteractionHandler, typename PhotonGenerator, typename Scorer, typename StepLength, typename Rng>
void RunMC(const cudaTextureObject_t& densityVolume,      //3D float
           const cudaTextureObject_t& materialTypeVolume, //3D uint8_t
           const CudaParameters& c_param,
           const unsigned numberOfThreads,
           InteractionHandler interaction,
           PhotonGenerator photonGen,
           Scorer scorer,
           StepLength stepLength,
           Rng rng) {
    //Calculate maximum occupancy settings
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
                     // maximum occupancy for a full device launch
    int gridSize;    // The actual grid size needed, based on input size

    CUDA_SAFE_CALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize,
                                                      &blockSize,
                                                      monteCarloKernel<PhotonGenerator, InteractionHandler, Scorer, StepLength, Rng>,
                                                      0, //no dynamic shared memory use
                                                      numberOfThreads));
    // Round up according to array size
    gridSize = (numberOfThreads + blockSize - 1) / blockSize;
    // Launch the kernel
    cudaProfilerStart();
    monteCarloKernel << <gridSize, blockSize>>> (densityVolume,
                                                 materialTypeVolume,
                                                 c_param,
                                                 numberOfThreads,
                                                 interaction,
                                                 photonGen,
                                                 scorer,
                                                 stepLength,
                                                 rng);
    cudaProfilerStop();
    CUDA_KERNEL_ERRCHECK;
};
}
}
