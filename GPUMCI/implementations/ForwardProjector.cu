#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <memory>
#include <iostream>
#include <algorithm>

#include <odl_cpp_utils/cuda/disableThrustWarnings.h>
#include <thrust/device_vector.h>
#include <odl_cpp_utils/cuda/enableThrustWarnings.h>

#include <GPUMCI/implementations/ForwardProjector.h>

#include <GPUMCI/physics/MaterialEntry.h>
#include <GPUMCI/physics/CudaSettings.h>
#include <GPUMCI/implementations/MaterialUtils.cuh>
#include <GPUMCI/implementations/WoodcockUtils.cuh>

//CudaMonteCarlo parts
#include <GPUMCI/detector/DetectorCBCT.cuh>
#include <GPUMCI/photongenerator/PhotonGeneratorUniform.cuh>
#include <GPUMCI/rng/DeterministicRng.cuh>
#include <GPUMCI/interactions/PhotonPhoto.cuh>
#include <GPUMCI/interactions/InteractionHandlerAttenuating.cuh>
#include <GPUMCI/interactions/FixedStep.cuh>

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
struct ForwardProjectorCuData {
    ForwardProjectorCuData(const int3 volumeSize,
                           const int2 detectorSize,
                           const MaterialData& attenuationData_)
        : attenuationData(attenuationData_),
          densityVolume(std::make_shared<BoundTexture3D<float>>(volumeSize,
                                                                cudaAddressModeClamp,
                                                                cudaFilterModeLinear,
                                                                cudaReadModeElementType)),
          materialTypeVolume(std::make_shared<BoundTexture3D<uint8_t>>(volumeSize,
                                                                       cudaAddressModeClamp,
                                                                       cudaFilterModePoint,
                                                                       cudaReadModeElementType)) {

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
    ForwardProjectorCuData(const ForwardProjectorCuData&) = delete;
    ForwardProjectorCuData& operator=(const ForwardProjectorCuData&) = delete;

    const MaterialData attenuationData;
    std::shared_ptr<BoundTexture3D<float>> densityVolume;
    std::shared_ptr<BoundTexture3D<uint8_t>> materialTypeVolume;
    std::shared_ptr<BoundTexture2D<float4>> texMaterial;
};
}

ForwardProjector::ForwardProjector(const Eigen::Vector3i& volumeSize,
                                   const Eigen::Vector3d& volumeOrigin,
                                   const Eigen::Vector3d& voxelSize,
                                   const Eigen::Vector2i& detectorSize,
                                   const MaterialData& attenuationData)
    : _param{volumeSize, volumeOrigin, voxelSize, attenuationData.energyStep},
      _detectorSize(detectorSize) {
    // Initialize the cuda side
    _cudaData = std::make_shared<cuda::ForwardProjectorCuData>(make_int3(volumeSize),
                                                               make_int2(detectorSize),
                                                               attenuationData);
}

void ForwardProjector::setData(const float* densityDevice,
                               const uint8_t* materialTypeDevice) {
    //Set the density and materials
    _cudaData->densityVolume->setData(densityDevice);
    _cudaData->materialTypeVolume->setData(materialTypeDevice);
}

void ForwardProjector::project(const Eigen::Vector3d& sourcePosition,
                               const Eigen::Vector3d& detectorOrigin,
                               const Eigen::Vector3d& pixelDirectionU,
                               const Eigen::Vector3d& pixelDirectionV,
                               const float energy,
                               const float stepLength,
                               float* primary) const {
    // Setup kernel configuration
    unsigned numberOfThreads = cuda::nThreads(make_int2(_detectorSize));
    float2 inversePixelSize = make_float2(1.0f / (float)pixelDirectionU.norm(),
                                          1.0f / (float)pixelDirectionV.norm());

    // Create a detector
    cuda::DetectorCBCT detector{make_float3(detectorOrigin),
                                make_float3(pixelDirectionU),
                                make_float3(pixelDirectionV),
                                inversePixelSize,
                                make_int2(_detectorSize),
                                narrow_cast<unsigned>(_detectorSize[0]),
                                primary};

    //Use a analytic photon generator
    cuda::PhotonGeneratorUniform photonGenerator{make_int2(_detectorSize),
                                                 make_float3(detectorOrigin),
                                                 make_float3(pixelDirectionU),
                                                 make_float3(pixelDirectionV),
                                                 make_float3(sourcePosition),
                                                 1, //deterministic, so we only need one run
                                                 energy,
                                                 0.0f};

    //Simple interaction handler
    cuda::InteractionHandlerAttenuating interaction{_cudaData->texMaterial->tex()};

    cuda::FixedStep step{stepLength};
    cuda::DeterministicRng rng{};

    cuda::RunMC(_cudaData->densityVolume->tex(),
                _cudaData->materialTypeVolume->tex(),
                _param,
                numberOfThreads,
                interaction,
                photonGenerator,
                detector,
                step,
                rng);
}
}
