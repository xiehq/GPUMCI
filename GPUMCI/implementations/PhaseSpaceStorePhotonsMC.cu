#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <memory>

#include <odl_cpp_utils/cuda/disableThrustWarnings.h>
#include <odl_cpp_utils/cuda/enableThrustWarnings.h>
#include <thrust/device_vector.h>

#include <GPUMCI/implementations/PhaseSpaceStorePhotonsMC.h>

#include <GPUMCI/implementations/MaterialUtils.cuh>
#include <GPUMCI/physics/CudaSettings.h>
#include <GPUMCI/physics/MaterialEntry.h>

// CudaMonteCarlo parts
#include <GPUMCI/detector/DetectorStorePhoton.cuh>
#include <GPUMCI/interactions/ComptonPrecomputed.cuh>
#include <GPUMCI/interactions/InteractionHandlerPhoton.cuh>
#include <GPUMCI/interactions/PhotonPhoto.cuh>
#include <GPUMCI/interactions/RayleighPrecomputed.cuh>
#include <GPUMCI/interactions/WoodcockStep.cuh>
#include <GPUMCI/photongenerator/PhotonGeneratorPhaseSpace.cuh>
#include <GPUMCI/rng/CurandRng.cuh>

#include <GPUMCI/physics/CudaMonteCarlo.cuh>

#include <odl_cpp_utils/cuda/texture.h>
#include <odl_cpp_utils/utils/cast.h>

namespace gpumci {
namespace cuda {
namespace {
unsigned nThreads() {
    return 100000;
}
} // namespace

// Struct that holds all data needed for the cuda MC simulation
struct PhaseSpaceStorePhotonsMCCuData {
    PhaseSpaceStorePhotonsMCCuData(const int3 volumeSize,
                                   const MaterialData& attenuationData_,
                                   const InteractionTables& rayleighTables,
                                   const InteractionTables& comptonTables)
        : attenuationData(attenuationData_),
          densityVolume(std::make_shared<BoundTexture3D<float>>(volumeSize,
                                                                cudaAddressModeClamp,
                                                                cudaFilterModeLinear,
                                                                cudaReadModeElementType)),
          materialTypeVolume(std::make_shared<BoundTexture3D<uint8_t>>(volumeSize,
                                                                       cudaAddressModeClamp,
                                                                       cudaFilterModePoint,
                                                                       cudaReadModeElementType)),
          rng(nThreads()),
          rayleigh(rayleighTables),
          compton(comptonTables) {
        int n_energy = narrow_cast<int>(attenuationData.n_energies);
        int n_materials = narrow_cast<int>(attenuationData.n_materials);

        // Interaction
        thrust::device_vector<float4> data = util::make_material_device(attenuationData);
        texMaterial = std::make_shared<BoundTexture2D<float4>>(int2{n_energy, n_materials},
                                                               cudaAddressModeClamp,
                                                               cudaFilterModeLinear,
                                                               cudaReadModeElementType);
        texMaterial->setData(thrust::raw_pointer_cast(&data[0]));
    }

    // Nocopy
    PhaseSpaceStorePhotonsMCCuData(const PhaseSpaceStorePhotonsMCCuData&) = delete;
    PhaseSpaceStorePhotonsMCCuData& operator=(const PhaseSpaceStorePhotonsMCCuData&) = delete;

    const MaterialData attenuationData;
    std::shared_ptr<BoundTexture3D<float>> densityVolume;
    std::shared_ptr<BoundTexture3D<uint8_t>> materialTypeVolume;
    std::shared_ptr<BoundTexture2D<float4>> texMaterial;
    std::shared_ptr<WoodcockStep> woodcockStep;
    const RayleighPrecomputed rayleigh;
    const ComptonPrecomputed compton;
    curandRng rng;
};
} // namespace cuda

PhaseSpaceStorePhotonsMC::PhaseSpaceStorePhotonsMC(const Eigen::Vector3i& volumeSize,
                                                   const Eigen::Vector3d& volumeOrigin,
                                                   const Eigen::Vector3d& voxelSize,
                                                   const MaterialData& attenuationData,
                                                   const InteractionTables& rayleighTables,
                                                   const InteractionTables& comptonTables)
    : _param{volumeSize, volumeOrigin, voxelSize, attenuationData.energyStep} {
    // Initialize the cuda side
    _cudaData = std::make_shared<cuda::PhaseSpaceStorePhotonsMCCuData>(make_int3(volumeSize),
                                                                       attenuationData,
                                                                       rayleighTables,
                                                                       comptonTables);
}

void PhaseSpaceStorePhotonsMC::setData(const float* densityDevice,
                                       const uint8_t* materialTypeDevice) {
    // Set the density and materials
    _cudaData->densityVolume->setData(densityDevice);
    _cudaData->materialTypeVolume->setData(materialTypeDevice);

    // Since the densities have updated, we need to update the woodcock table
    int n_energy = narrow_cast<int>(_cudaData->attenuationData.n_energies);
    _cudaData->woodcockStep = std::make_shared<cuda::WoodcockStep>(densityDevice,
                                                                   materialTypeDevice,
                                                                   _param.volumeSize,
                                                                   n_energy,
                                                                   _param.invEnergyStep,
                                                                   _cudaData->attenuationData);
}

/**/
void PhaseSpaceStorePhotonsMC::setData(const std::vector<float>& densityHost,
                           const std::vector<uint8_t>& materialHost) {
    thrust::device_vector<float> dDevice(densityHost);
    thrust::device_vector<uint8_t> mDevice(materialHost);
    setData(thrust::raw_pointer_cast(&dDevice[0]), thrust::raw_pointer_cast(&mDevice[0]));
}
/**/

void PhaseSpaceStorePhotonsMC::project(const std::vector<cuda::CudaMonteCarloParticle>& particles_in,
                                       std::vector<cuda::CudaMonteCarloParticle>& particles_out) const {
    // Setup kernel configuration
    unsigned numberOfThreads = cuda::nThreads();

    if (particles_in.size() != particles_out.size())
        throw std::invalid_argument("phase space sizes do not match");

    numberOfThreads = min(narrow_cast<unsigned>(particles_in.size()), numberOfThreads);
    // Create a detector
    cuda::DetectorStorePhotons<cuda::CudaMonteCarloParticle> detector{narrow_cast<unsigned>(particles_in.size()),
                                                                      numberOfThreads};

    // Use a phase space photon generator
    cuda::PhotonGeneratorPhaseSpace photonGenerator{particles_in, 1, numberOfThreads};

    // Simple interaction handler
    auto interaction = cuda::makePhotonInteractionHandler(_cudaData->compton.deviceSide(),
                                                          _cudaData->rayleigh.deviceSide(),
                                                          cuda::PhotonPhoto{},
                                                          _cudaData->texMaterial->tex());
    // Run montecarlo
    cuda::RunMC(_cudaData->densityVolume->tex(),
                _cudaData->materialTypeVolume->tex(),
                _param,
                numberOfThreads,
                interaction,
                photonGenerator.deviceSide(),
                detector.deviceSide(),
                _cudaData->woodcockStep->deviceSide(),
                _cudaData->rng.deviceSide());

    // Copy the results back to host side
    detector.copy_to_host(particles_out);
}
} // namespace gpumci
