#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <memory>

#include <odl_cpp_utils/cuda/disableThrustWarnings.h>
#include <odl_cpp_utils/cuda/enableThrustWarnings.h>
#include <thrust/device_vector.h>

#include <GPUMCI/implementations/PhaseSpaceMC.h>

#include <GPUMCI/implementations/MaterialUtils.cuh>
#include <GPUMCI/physics/CudaSettings.h>
#include <GPUMCI/physics/MaterialEntry.h>

// CudaMonteCarlo parts
#include <GPUMCI/detector/DetectorCBCTScatter.cuh>
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
unsigned nThreads(int2 detectorSize) {
    return detectorSize.x * detectorSize.y; // return 100000; //gjb
}
} // namespace

// Struct that holds all data needed for the cuda MC simulation
struct PhaseSpaceMCCuData {
    PhaseSpaceMCCuData(const int3 volumeSize,
                       const int2 detectorSize,
                       const unsigned& numThreads,
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
          rng(numThreads), // rng(nThreads(detectorSize)),
          rayleigh(rayleighTables),
          compton(comptonTables) { // detectorSize.x * detectorSize.y) {

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
    PhaseSpaceMCCuData(const PhaseSpaceMCCuData&) = delete;
    PhaseSpaceMCCuData& operator=(const PhaseSpaceMCCuData&) = delete;

    const MaterialData attenuationData;
    std::shared_ptr<BoundTexture3D<float>> densityVolume; // row major order data
    std::shared_ptr<BoundTexture3D<uint8_t>> materialTypeVolume;
    std::shared_ptr<BoundTexture2D<float4>> texMaterial;
    std::shared_ptr<WoodcockStep> woodcockStep;
    const RayleighPrecomputed rayleigh;
    const ComptonPrecomputed compton;
    curandRng rng;
};
} // namespace cuda

/* modified constructor GJB* with ability to set num of threads*/
PhaseSpaceMC::PhaseSpaceMC(const Eigen::Vector3i& volumeSize,
                           const Eigen::Vector3d& volumeOrigin,
                           const Eigen::Vector3d& voxelSize,
                           const Eigen::Vector2i& detectorSize,
                           unsigned n_runs,
                           unsigned n_threads,
                           const MaterialData& attenuationData,
                           const InteractionTables& rayleighTables,
                           const InteractionTables& comptonTables)
    : _param{volumeSize, volumeOrigin, voxelSize, attenuationData.energyStep},
      _detectorSize(detectorSize),
      _nThreads(n_threads),
      _nRuns(n_runs) {
    // Initialize the cuda side
    _cudaData = std::make_shared<cuda::PhaseSpaceMCCuData>(make_int3(volumeSize),
                                                           make_int2(detectorSize),
                                                           n_threads,
                                                           attenuationData,
                                                           rayleighTables,
                                                           comptonTables);
}

/* Original Constructor*/
PhaseSpaceMC::PhaseSpaceMC(const Eigen::Vector3i& volumeSize,
                           const Eigen::Vector3d& volumeOrigin,
                           const Eigen::Vector3d& voxelSize,
                           const Eigen::Vector2i& detectorSize,
                           unsigned n_runs,
                           const MaterialData& attenuationData,
                           const InteractionTables& rayleighTables,
                           const InteractionTables& comptonTables)
    : _param{volumeSize, volumeOrigin, voxelSize, attenuationData.energyStep},
      _detectorSize(detectorSize),
      _nThreads(cuda::nThreads(make_int2(_detectorSize))),
      _nRuns(n_runs) {
    // Initialize the cuda side
    _cudaData = std::make_shared<cuda::PhaseSpaceMCCuData>(make_int3(volumeSize),
                                                           make_int2(detectorSize),
                                                           _nThreads,
                                                           attenuationData,
                                                           rayleighTables,
                                                           comptonTables);
}

void PhaseSpaceMC::setData(const float* densityDevice,
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

void PhaseSpaceMC::setData(const std::vector<float>& densityHost,
                           const std::vector<uint8_t>& materialHost) {
    thrust::device_vector<float> dDevice(densityHost);
    thrust::device_vector<uint8_t> mDevice(materialHost);
    setData(thrust::raw_pointer_cast(&dDevice[0]), thrust::raw_pointer_cast(&mDevice[0]));
}

void PhaseSpaceMC::project(const Eigen::Vector3d& sourcePosition,
                           const Eigen::Vector3d& detectorOrigin,
                           const Eigen::Vector3d& pixelDirectionU,
                           const Eigen::Vector3d& pixelDirectionV,
                           const std::vector<cuda::CudaMonteCarloParticle>& particles,
                           float* primary,
                           float* scatter) const {
    // Setup kernel configuration

    /*
    if (particles.size() > numberOfThreads)
        throw std::invalid_argument("phase space to large, allocated to little rng");
    else
        numberOfThreads = narrow_cast<unsigned>(particles.size());
    */
    unsigned numberOfThreads = _nThreads;
    numberOfThreads = min(narrow_cast<unsigned>(particles.size()), numberOfThreads);

    float2 inversePixelSize = make_float2(1.0f / (float)pixelDirectionU.norm(),
                                          1.0f / (float)pixelDirectionV.norm());

    // Create a detector
    cuda::DetectorCBCTScatter detector{make_float3(detectorOrigin),
                                       make_float3(pixelDirectionU),
                                       make_float3(pixelDirectionV),
                                       inversePixelSize,
                                       make_int2(_detectorSize),
                                       narrow_cast<unsigned>(_detectorSize[0]),
                                       primary,
                                       scatter};

    // Use a phase space photon generator
    cuda::PhotonGeneratorPhaseSpace photonGenerator{particles, _nRuns, numberOfThreads};

    // Simple interaction handler
    auto interaction = cuda::makePhotonInteractionHandler(_cudaData->compton.deviceSide(),
                                                          _cudaData->rayleigh.deviceSide(),
                                                          cuda::PhotonPhoto{},
                                                          _cudaData->texMaterial->tex());
    cuda::RunMC(_cudaData->densityVolume->tex(),
                _cudaData->materialTypeVolume->tex(),
                _param,
                numberOfThreads,
                interaction,
                photonGenerator.deviceSide(),
                detector,
                _cudaData->woodcockStep->deviceSide(),
                _cudaData->rng.deviceSide());
}
void PhaseSpaceMC::project(const Eigen::Vector3d& sourcePosition,
                           const Eigen::Vector3d& detectorOrigin,
                           const Eigen::Vector3d& pixelDirectionU,
                           const Eigen::Vector3d& pixelDirectionV,
                           const std::vector<cuda::CudaMonteCarloParticle>& particles,
                           float* primary,
                           float* scatter,
                           float* secondary) const {
    // Setup kernel configuration

    /*
    if (particles.size() > numberOfThreads)
        throw std::invalid_argument("phase space to large, allocated to little rng");
    else
        numberOfThreads = narrow_cast<unsigned>(particles.size());
    */
    unsigned numberOfThreads = _nThreads;
    numberOfThreads = min(narrow_cast<unsigned>(particles.size()), numberOfThreads);

    float2 inversePixelSize = make_float2(1.0f / (float)pixelDirectionU.norm(),
                                          1.0f / (float)pixelDirectionV.norm());

    // Create a detector
    cuda::DetectorCBCTScatter detector{make_float3(detectorOrigin),
                                       make_float3(pixelDirectionU),
                                       make_float3(pixelDirectionV),
                                       inversePixelSize,
                                       make_int2(_detectorSize),
                                       narrow_cast<unsigned>(_detectorSize[0]),
                                       primary,
                                       scatter,
                                       secondary};

    // Use a phase space photon generator
    cuda::PhotonGeneratorPhaseSpace photonGenerator{particles, _nRuns, numberOfThreads};

    // Simple interaction handler
    auto interaction = cuda::makePhotonInteractionHandler(_cudaData->compton.deviceSide(),
                                                          _cudaData->rayleigh.deviceSide(),
                                                          cuda::PhotonPhoto{},
                                                          _cudaData->texMaterial->tex());
    cuda::RunMC(_cudaData->densityVolume->tex(),
                _cudaData->materialTypeVolume->tex(),
                _param,
                numberOfThreads,
                interaction,
                photonGenerator.deviceSide(),
                detector,
                _cudaData->woodcockStep->deviceSide(),
                _cudaData->rng.deviceSide());
}
} // namespace gpumci
