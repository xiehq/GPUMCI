#pragma once

#include <GPUMCI/implementations/WoodcockUtils.cuh>
#include <GPUMCI/physics/CudaSettings.h>
#include <odl_cpp_utils/cuda/texture.h>

namespace gpumci {
namespace cuda {
/**
	*/
struct FixedStep {
    FixedStep(const float step_mm)
        : _step_cm(step_mm / 10.0f) {}

    FixedStep(const FixedStep&) = default;

    template <typename Rng, typename Particle>
    __device__ void operator()(const Particle& myPhoton, float& expected_value_cm, float& step_cm, Rng& rng) {
        expected_value_cm = _step_cm;
        step_cm = _step_cm;
    }

  private:
    const float _step_cm;
};
}
}
