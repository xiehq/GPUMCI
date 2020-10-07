#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <memory>
#include <iostream>
#include <algorithm>

#include <odl_cpp_utils/cuda/disableThrustWarnings.h>
#include <thrust/device_vector.h>
#include <odl_cpp_utils/cuda/enableThrustWarnings.h>

#include <GPUMCI/implementations/DoseMC.h>

#include <GPUMCI/physics/MaterialEntry.h>
#include <GPUMCI/physics/CudaSettings.h>
#include <GPUMCI/implementations/MaterialUtils.cuh>
#include <GPUMCI/implementations/WoodcockUtils.cuh>

//CudaMonteCarlo parts
#include <GPUMCI/detector/DetectorCBCTScatter.cuh>
#include <GPUMCI/detector/DoseDetector.cuh>
#include <GPUMCI/photongenerator/PhotonGeneratorUniform.cuh>
#include <GPUMCI/rng/CurandRng.cuh>
#include <GPUMCI/interactions/PhotonPhoto.cuh>
#include <GPUMCI/interactions/InteractionHandlerPhoton.cuh>
#include <GPUMCI/interactions/ComptonPrecomputed.cuh>
#include <GPUMCI/interactions/RayleighPrecomputed.cuh>
#include <GPUMCI/interactions/WoodcockStep.cuh>

#include <GPUMCI/physics/CudaMonteCarlo.cuh>

#include <odl_cpp_utils/utils/cast.h>
#include <odl_cpp_utils/cuda/texture.h>

namespace gpumci {
namespace cuda {
namespace {
unsigned nThreads(int2 detectorSize) {
    return detectorSize.x * detectorSize.y;
}
}

//Struct that holds all data needed for the cuda MC simulation
struct DoseMCCuData {
    DoseMCCuData(const int3 volumeSize,
                 const int2 detectorSize_,
                 const MaterialData& attenuationData_,
                 const InteractionTables& rayleighTables,
                 const InteractionTables& comptonTables)
        : detectorSize(detectorSize_),
          attenuationData(attenuationData_),
          densityVolume(std::make_shared<BoundTexture3D<float>>(volumeSize,
                                                                cudaAddressModeClamp,
                                                                cudaFilterModeLinear,
                                                                cudaReadModeElementType)),
          materialTypeVolume(std::make_shared<BoundTexture3D<uint8_t>>(volumeSize,
                                                                       cudaAddressModeClamp,
                                                                       cudaFilterModePoint,
                                                                       cudaReadModeElementType)),
          rng(nThreads(detectorSize)),
          rayleigh(rayleighTables),
          compton(comptonTables) { //detectorSize.x * detectorSize.y) {

        int n_energy = narrow_cast<int>(attenuationData.n_energies);
        int n_materials = narrow_cast<int>(attenuationData.n_materials);

        //Interaction
        thrust::device_vector<float4> data = util::make_material_device(attenuationData);
        texMaterial = std::make_shared<BoundTexture2D<float4>>(int2{n_energy, n_materials},
                                                               cudaAddressModeClamp,
                                                               cudaFilterModeLinear,
                                                               cudaReadModeElementType);
        texMaterial->setData(thrust::raw_pointer_cast(&data[0]));
    }

    //Nocopy
    DoseMCCuData(const DoseMCCuData&) = delete;
    DoseMCCuData& operator=(const DoseMCCuData&) = delete;

    const int2 detectorSize;
    const MaterialData attenuationData;
    std::shared_ptr<BoundTexture3D<float>> densityVolume;
    std::shared_ptr<BoundTexture3D<uint8_t>> materialTypeVolume;
    std::shared_ptr<BoundTexture2D<float4>> texMaterial;
    std::shared_ptr<WoodcockStep> woodcockStep;
    const RayleighPrecomputed rayleigh;
    const ComptonPrecomputed compton;
    curandRng rng;
};
}
DoseMC::DoseMC(const Eigen::Vector3i& volumeSize,
               const Eigen::Vector3d& volumeOrigin,
               const Eigen::Vector3d& voxelSize,
               const Eigen::Vector2i& detectorSize,
               int n_runs,
               const MaterialData& attenuationData,
               const InteractionTables& rayleighTables,
               const InteractionTables& comptonTables)
    : _param{volumeSize, volumeOrigin, voxelSize, attenuationData.energyStep},
      _nRuns(n_runs) {
    // Initialize the cuda side
    _cudaData = std::make_shared<cuda::DoseMCCuData>(make_int3(volumeSize),
                                                     make_int2(detectorSize),
                                                     attenuationData,
                                                     rayleighTables,
                                                     comptonTables);
}

void DoseMC::setData(const float* densityDevice,
                     const uint8_t* materialTypeDevice) {
    //Set the density and materials
    _cudaData->densityVolume->setData(densityDevice);
    _cudaData->materialTypeVolume->setData(materialTypeDevice);

    //Since the densities have updated, we need to update the woodcock table
    int n_energy = narrow_cast<int>(_cudaData->attenuationData.n_energies);
    _cudaData->woodcockStep = std::make_shared<cuda::WoodcockStep>(densityDevice,
                                                                   materialTypeDevice,
                                                                   _param.volumeSize,
                                                                   n_energy,
                                                                   _param.invEnergyStep,
                                                                   _cudaData->attenuationData);
}

void DoseMC::project(const Eigen::Vector3d& sourcePosition,
                     const Eigen::Vector3d& detectorOrigin,
                     const Eigen::Vector3d& pixelDirectionU,
                     const Eigen::Vector3d& pixelDirectionV,
                     float* primary,
                     float* scatter,
                     float* dose_volume) const {
    // Setup kernel configuration
    unsigned numberOfThreads = cuda::nThreads(_cudaData->detectorSize);
    float2 inversePixelSize = make_float2(1.0f / (float)pixelDirectionU.norm(),
                                          1.0f / (float)pixelDirectionV.norm());

    // Create a detector
    cuda::DetectorCBCTScatter detector{make_float3(detectorOrigin),
                                       make_float3(pixelDirectionU),
                                       make_float3(pixelDirectionV),
                                       inversePixelSize,
                                       _cudaData->detectorSize,
                                       narrow_cast<unsigned>(_cudaData->detectorSize.x),
                                       primary,
                                       scatter};

    //Use a analytic photon generator
    cuda::PhotonGeneratorUniform photonGenerator{_cudaData->detectorSize,
                                                 make_float3(detectorOrigin),
                                                 make_float3(pixelDirectionU),
                                                 make_float3(pixelDirectionV),
                                                 make_float3(sourcePosition),
                                                 _nRuns};

    //Simple interaction handler
    auto interaction = cuda::makePhotonInteractionHandler(_cudaData->compton.deviceSide(),
                                                          _cudaData->rayleigh.deviceSide(),
                                                          cuda::PhotonPhoto{},
                                                          _cudaData->texMaterial->tex());

    cuda::DoseDetector<decltype(interaction)> dose_interaction{_param.volumeMin,
                                                               _param.inverseVoxelSize,
                                                               _param.volumeSize,
                                                               dose_volume,
                                                               interaction};

    RunMC(_cudaData->densityVolume->tex(),
          _cudaData->materialTypeVolume->tex(),
          _param,
          numberOfThreads,
          dose_interaction,
          photonGenerator,
          detector,
          _cudaData->woodcockStep->deviceSide(),
          _cudaData->rng.deviceSide());
}
}
