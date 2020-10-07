#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <memory>
#include <iostream>

#include <odl_cpp_utils/cuda/disableThrustWarnings.h>
#include <thrust/device_vector.h>
#include <odl_cpp_utils/cuda/enableThrustWarnings.h>

#include <GPUMCI/implementations/SimpleMC.h>

#include <GPUMCI/physics/CudaSettings.h>
#include <GPUMCI/detector/DetectorCBCTScatter.cuh>
#include <GPUMCI/photongenerator/PhotonGeneratorUniform.cuh>
#include <GPUMCI/rng/MWCRng.cuh>
#include <GPUMCI/interactions/PhotonPhoto.cuh>
#include <GPUMCI/interactions/InteractionHandlerPhoton.cuh>
#include <GPUMCI/interactions/ComptonEverett.cuh>
#include <GPUMCI/interactions/NoInteraction.cuh>
#include <GPUMCI/interactions/FixedStep.cuh>

#include <GPUMCI/physics/CudaMonteCarlo.cuh>

#include <odl_cpp_utils/utils/cast.h>
#include <odl_cpp_utils/cuda/texture.h>

namespace gpumci {
namespace cuda {

struct SimpleMCCuData {

    SimpleMCCuData(const int3 volumeSize,
                   const int2 detectorSize)
        : densityVolume(std::make_shared<BoundTexture3D<float>>(volumeSize,
                                                                cudaAddressModeClamp,
                                                                cudaFilterModeLinear,
                                                                cudaReadModeElementType)),
          materialTypeVolume(std::make_shared<BoundTexture3D<uint8_t>>(volumeSize,
                                                                       cudaAddressModeClamp,
                                                                       cudaFilterModePoint,
                                                                       cudaReadModeElementType)),
          rng(detectorSize.x * detectorSize.y) {

        int n_energy = 1; //Rely on clamping

        float compton = 0.5f;
        float ray = 0.0f;
        float photo = 0.0f;
        int n_materials = 2;
        //Interaction
        thrust::device_vector<float4> data(n_energy * n_materials, make_float4(compton, ray, photo, 0.0f));
        texMaterial = std::make_shared<BoundTexture2D<float4>>(int2{n_energy, n_materials},
                                                               cudaAddressModeClamp,
                                                               cudaFilterModeLinear,
                                                               cudaReadModeElementType);
        texMaterial->setData(thrust::raw_pointer_cast(&data[0]));

        //Todo scale by density
        step = std::make_shared<cuda::FixedStep>(1.0f / (compton + ray + photo));
    }
    SimpleMCCuData(const SimpleMCCuData&) = delete;
    SimpleMCCuData& operator=(const SimpleMCCuData&) = delete;

    std::shared_ptr<BoundTexture3D<float>> densityVolume;
    std::shared_ptr<BoundTexture3D<uint8_t>> materialTypeVolume;
    std::shared_ptr<PhotonGeneratorUniform> photonGenerator;
    std::shared_ptr<BoundTexture2D<float4>> texMaterial;
    std::shared_ptr<cuda::FixedStep> step;
    MWCRng rng;
};
}
SimpleMC::SimpleMC(const Eigen::Vector3i& volumeSize,
                   const Eigen::Vector3d& volumeOrigin,
                   const Eigen::Vector3d& voxelSize,
                   const Eigen::Vector2i& detectorSize)
    : _param{volumeSize, volumeOrigin, voxelSize, 0.01},
      _volumeSize(volumeSize),
      _volumeOrigin(volumeOrigin),
      _voxelSize(voxelSize),
      _detectorSize(detectorSize) {
    _cudaData = std::make_shared<cuda::SimpleMCCuData>(make_int3(volumeSize),
                                                       make_int2(detectorSize));
}

void SimpleMC::setData(const float* densityDevice,
                       const uint8_t* materialTypeDevice) {
    _cudaData->densityVolume->setData(densityDevice);
    _cudaData->materialTypeVolume->setData(materialTypeDevice);
}

void SimpleMC::project(const Eigen::Vector3d& sourcePosition,
                       const Eigen::Vector3d& detectorOrigin,
                       const Eigen::Vector3d& pixelDirectionU,
                       const Eigen::Vector3d& pixelDirectionV,
                       float* primary,
                       float* scatter) const {
    // Setup kernel configuration
    unsigned numberOfThreads = _detectorSize[0] * _detectorSize[1];

    float2 inversePixelSize = make_float2(1.0f / (float)pixelDirectionU.norm(),
                                          1.0f / (float)pixelDirectionV.norm());

    cuda::DetectorCBCTScatter detector{make_float3(detectorOrigin),
                                       make_float3(pixelDirectionU),
                                       make_float3(pixelDirectionV),
                                       inversePixelSize,
                                       make_int2(_detectorSize),
                                       narrow_cast<unsigned>(_detectorSize[0]),
                                       primary,
                                       scatter};
    cuda::PhotonGeneratorUniform photonGenerator{make_int2(_detectorSize),
                                                 make_float3(detectorOrigin),
                                                 make_float3(pixelDirectionU),
                                                 make_float3(pixelDirectionV),
                                                 make_float3(sourcePosition),
                                                 100};
    auto interaction = cuda::makePhotonInteractionHandler(cuda::ComptonEverett{},
                                                          cuda::NoInteraction{},
                                                          cuda::PhotonPhoto{},
                                                          _cudaData->texMaterial->tex());
    cuda::RunMC(_cudaData->densityVolume->tex(),
                _cudaData->materialTypeVolume->tex(),
                _param,
                numberOfThreads,
                interaction,
                photonGenerator,
                detector,
                *_cudaData->step,
                _cudaData->rng.deviceSide());
}
}
