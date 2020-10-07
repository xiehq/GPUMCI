#pragma once

#include <GPUMCI/physics/CudaSettings.h>
#include <GPUMCI/interactions/PrecomputedInteractionUtils.cuh>
#include <GPUMCI/utils/physical_constants.h>

#include <odl_cpp_utils/cuda/disableThrustWarnings.h>
#include <thrust/device_vector.h>
#include <odl_cpp_utils/cuda/enableThrustWarnings.h>

#include <odl_cpp_utils/utils/cast.h>
#include <odl_cpp_utils/cuda/texture.h>

namespace gpumci {
namespace cuda {
struct ComptonPrecomputedDevice {
    __host__ ComptonPrecomputedDevice(const cudaTextureObject_t& texComptonTables,
                                      const float interactionMinEnergy,
                                      const float interactionInvEnergyStep,
                                      const float interactionTableWidth)
        : _texComptonTables(texComptonTables),
          _interactionMinEnergy(interactionMinEnergy),
          _interactionInvEnergyStep(interactionInvEnergyStep),
          _interactionTableWidth(interactionTableWidth) {}

    template <typename Rng, typename Particle>
    __device__ void operator()(const int myMedium, const Particle& myPhoton, float& costheta, float& energy, Rng& rng) const {
        // The polar angle is sampled in a table that for each material and energy tabulates cos(theta).
        // The table contains the culumative cross section as function of cos(theta).
        // By generating a random number belonging to [0,1], cos(theta) is determined.

        const float tableIndex = 0.5f + _interactionTableWidth * rng.rand();
        const float energyIndex = 0.5f + (myPhoton.energy - _interactionMinEnergy) * _interactionInvEnergyStep;

        costheta = tex2DLayered<float>(_texComptonTables, tableIndex, energyIndex, myMedium);

        //Calculate energy after interaction
        const float k = myPhoton.energy * INVERSE_ELECTRON_REST_ENERGY_MEV;
        const float x = 1.0f / (1.0f + k * (1.0f - costheta));

        //        printf("%f %f %f %f %f\n", myPhoton.energy, costheta, x, tableIndex, energyIndex);
        energy = myPhoton.energy * x;
    }

  private:
    const cudaTextureObject_t _texComptonTables;
    const float _interactionMinEnergy;
    const float _interactionInvEnergyStep;
    const float _interactionTableWidth;
};
/**
	Method with computes a Compton Interaction by randomly sampling the cosine of the angle from a lookup table.
	*/

struct ComptonPrecomputed {
    ComptonPrecomputed(const InteractionTables& comptonTables)
        : _invEnergyStep(static_cast<float>(1.0 / comptonTables.energyStep)),
          _interactionTableWidth(static_cast<float>(comptonTables.n_samples - 1)) {
        int n_energy = narrow_cast<int>(comptonTables.n_energies);
        int n_materials = narrow_cast<int>(comptonTables.n_materials);
        int n_samples = narrow_cast<int>(comptonTables.n_samples);

        _comptonTables = std::make_shared<BoundTexture3D<float, cudaArrayLayered>>(int3{n_samples, n_energy, n_materials},
                                                                                   cudaAddressModeClamp,
                                                                                   cudaFilterModeLinear,
                                                                                   cudaReadModeElementType);

        thrust::device_vector<float> comptonData = util::make_interaction_device(comptonTables);
        _comptonTables->setData(thrust::raw_pointer_cast(&comptonData[0]));
    }

    ComptonPrecomputedDevice deviceSide() const {
        return {_comptonTables->tex(), 0.0f, _invEnergyStep, _interactionTableWidth};
    }

  private:
    float _invEnergyStep;
    float _interactionTableWidth;
    std::shared_ptr<BoundTexture3D<float, cudaArrayLayered>> _comptonTables;
};
}
}
