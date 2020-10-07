#pragma once

#include <GPUMCI/implementations/WoodcockUtils.cuh>
#include <GPUMCI/physics/CudaSettings.h>
#include <odl_cpp_utils/cuda/texture.h>

namespace gpumci {
namespace cuda {
/**
 * Object that calculates the steplength using a woodcock lookup table
 */
struct WoodcockStepDevice {
    WoodcockStepDevice(const float invEnergyStep,
                       const cudaTextureObject_t texWoodcockStepLength)
        : _invEnergyStep(invEnergyStep),
          _texWoodcockStepLength(texWoodcockStepLength) {}

    //Find a step length
    template <typename Rng, typename Particle>
    __device__ void operator()(const Particle& myPhoton, float& expected_value_cm, float& step_cm, Rng& rng) {
        //Offset index by 0.5 due to interpolating cuda texture
        const float index = 0.5f + myPhoton.energy * _invEnergyStep;

        //Get the expected step length
        expected_value_cm = tex1D<float>(_texWoodcockStepLength, index);

        //Randomly select a step from an exponential distribution with mean expected_value_cm
        step_cm = -logf(rng.rand()) * expected_value_cm;
    }

  private:
    const float _invEnergyStep;
    const cudaTextureObject_t _texWoodcockStepLength;
};

struct WoodcockStep {
    WoodcockStep(const float* densityDevice,
                 const uint8_t* materialTypeDevice,
                 const int3 volumeSize,
                 const int n_energy,
                 const float invEnergyStep,
                 const MaterialData& attenuationData)
        : _invEnergyStep(invEnergyStep) {
        auto woodcock = cuda::util::make_woodcock_tables(densityDevice,
                                                         materialTypeDevice,
                                                         volumeSize,
                                                         attenuationData);
        _woodcockTable = std::make_shared<BoundTexture1D<float>>(n_energy,
                                                                 cudaAddressModeClamp,
                                                                 cudaFilterModeLinear,
                                                                 cudaReadModeElementType);
        _woodcockTable->setData(thrust::raw_pointer_cast(&woodcock[0]));
    }

    //make nocopy
    WoodcockStep(const WoodcockStep&) = delete;
    WoodcockStep& operator=(const WoodcockStep&) = delete;

    //Get the device side data holder
    WoodcockStepDevice deviceSide() const {
        return {_invEnergyStep, _woodcockTable->tex()};
    }

  private:
    const float _invEnergyStep;
    std::shared_ptr<BoundTexture1D<float>> _woodcockTable;
};
}
}
