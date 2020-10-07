#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

#include <odl_cpp_utils/cuda/disableThrustWarnings.h>
#include <thrust/device_vector.h>
#include <odl_cpp_utils/cuda/enableThrustWarnings.h>

#include <GPUMCI/implementations/AbsorbingMC.h>

#include <GPUMCI/rng/MWCRng.cuh>
#include <GPUMCI/physics/CudaSettings.h>
#include <GPUMCI/detector/DetectorCBCT.cuh>
#include <GPUMCI/photongenerator/PhotonGeneratorUniform.cuh>
#include <GPUMCI/interactions/InteractionHandlerProjector.cuh>
#include <GPUMCI/interactions/FixedStep.cuh>

#include <GPUMCI/physics/CudaMonteCarlo.cuh>

#include <odl_cpp_utils/utils/cast.h>
#include <odl_cpp_utils/cuda/texture.h>
#include <odl_cpp_utils/cuda/cutil_math.h>

namespace gpumci {
AbsorbingMC::AbsorbingMC(const Eigen::Vector3i& volumeSize,
                         const Eigen::Vector3d& volumeOrigin,
                         const Eigen::Vector3d& voxelSize,
                         const Eigen::Vector2i& detectorSize)
    : _volumeSize(volumeSize),
      _volumeOrigin(volumeOrigin),
      _voxelSize(voxelSize),
      _detectorSize(detectorSize) {
}

void AbsorbingMC::setData(const float* densityDevice,
                          const uint8_t* materialTypeDevice) {
    _densityDevice = densityDevice;
    _materialTypeDevice = materialTypeDevice;
}

void AbsorbingMC::project(const Eigen::Vector3d& sourcePosition,
                          const Eigen::Vector3d& detectorOrigin,
                          const Eigen::Vector3d& pixelDirectionU,
                          const Eigen::Vector3d& pixelDirectionV,
                          float* target) const {
    cuda::CudaParameters parameters = {_volumeSize, _volumeOrigin, _voxelSize, 0.01};

    BoundTexture3D<float> densityVolume{parameters.volumeSize,
                                        cudaAddressModeClamp,
                                        cudaFilterModeLinear,
                                        cudaReadModeElementType};
    densityVolume.setData(_densityDevice);
    BoundTexture3D<uint8_t> materialTypeVolume{parameters.volumeSize,
                                               cudaAddressModeClamp,
                                               cudaFilterModePoint,
                                               cudaReadModeElementType};
    materialTypeVolume.setData(_materialTypeDevice);

    float2 inversePixelSize = make_float2(1.0f / (float)pixelDirectionU.norm(),
                                          1.0f / (float)pixelDirectionV.norm());

    cuda::FixedStep step{1.0f};

    // Setup kernel configuration
    unsigned numberOfThreads = _detectorSize[0] * _detectorSize[1];

    cuda::DetectorCBCT detector{make_float3(detectorOrigin),
                                make_float3(pixelDirectionU),
                                make_float3(pixelDirectionV),
                                inversePixelSize,
                                make_int2(_detectorSize),
                                narrow_cast<unsigned>(_detectorSize[0]),
                                target};
    cuda::PhotonGeneratorUniform photonGenerator{make_int2(_detectorSize),
                                                 make_float3(detectorOrigin),
                                                 make_float3(pixelDirectionU),
                                                 make_float3(pixelDirectionV),
                                                 make_float3(sourcePosition),
                                                 100};

    cuda::InteractionHandlerProjector interaction{};

    cuda::MWCRng rng(numberOfThreads);

    RunMC(densityVolume.tex(),
          materialTypeVolume.tex(),
          parameters,
          numberOfThreads,
          interaction,
          photonGenerator,
          detector,
          step,
          rng.deviceSide());
}
}
