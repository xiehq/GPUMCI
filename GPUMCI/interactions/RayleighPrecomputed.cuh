#pragma once

#include <GPUMCI/physics/CudaSettings.h>
#include <GPUMCI/interactions/PrecomputedInteractionUtils.cuh>

#include <odl_cpp_utils/cuda/disableThrustWarnings.h>
#include <thrust/device_vector.h>
#include <odl_cpp_utils/cuda/enableThrustWarnings.h>

#include <odl_cpp_utils/utils/cast.h>
#include <odl_cpp_utils/cuda/texture.h>

namespace gpumci {
namespace cuda {
/**
	Method with computes a Rayleigh Interaction by randomly sampling the cosine of the angle from a lookup table.

	This method requires that the table `texRayleighTables` has been properly initialized.
	*/
struct RayleighPrecomputedDevice {
    __host__ RayleighPrecomputedDevice(const cudaTextureObject_t& texRayleighTables,
                                       const float interactionMinEnergy,
                                       const float interactionInvEnergyStep,
                                       const float interactionTableWidth)
        : _texRayleighTables(texRayleighTables),
          _interactionMinEnergy(interactionMinEnergy),
          _interactionInvEnergyStep(interactionInvEnergyStep),
          _interactionTableWidth(interactionTableWidth) {}

    template <typename Rng, typename Particle>
    __device__ void operator()(const int myMedium, const Particle& myPhoton, float& costheta, float& energy, Rng& rng) const {
        // Rayleigh scattering does not alter the energy of the photon.
        float energyIndex = 0.5f + (myPhoton.energy - _interactionMinEnergy) * _interactionInvEnergyStep;

        // The polar angle is sampled in a table that for each material and energy tabulates cos(theta).
        // The table contains the culumative cross section as function of cos(theta).
        // By generating a random number belonging to [0,1] cos(theta) is determined.
        const float tableIndex = 0.5f + _interactionTableWidth * rng.rand();

        costheta = tex2DLayered<float>(_texRayleighTables, tableIndex, energyIndex, myMedium);
    }

  private:
    const cudaTextureObject_t _texRayleighTables;
    const float _interactionMinEnergy;
    const float _interactionInvEnergyStep;
    const float _interactionTableWidth;
};

struct RayleighPrecomputed {
    RayleighPrecomputed(const InteractionTables& rayleighTables)
        : _invEnergyStep(static_cast<float>(1.0 / rayleighTables.energyStep)),
          _interactionTableWidth(static_cast<float>(rayleighTables.n_samples - 1)) {
        int n_energy_ray = narrow_cast<int>(rayleighTables.n_energies);
        int n_materials_ray = narrow_cast<int>(rayleighTables.n_materials);
        int n_samples_ray = narrow_cast<int>(rayleighTables.n_samples);

        _rayleighTables = std::make_shared<BoundTexture3D<float, cudaArrayLayered>>(int3{n_samples_ray, n_energy_ray, n_materials_ray},
                                                                                    cudaAddressModeClamp,
                                                                                    cudaFilterModeLinear,
                                                                                    cudaReadModeElementType);

        thrust::device_vector<float> rayleighData = util::make_interaction_device(rayleighTables);
        _rayleighTables->setData(thrust::raw_pointer_cast(&rayleighData[0]));
    }

    RayleighPrecomputedDevice deviceSide() const {
        return {_rayleighTables->tex(), 0.0f, _invEnergyStep, _interactionTableWidth};
    }

  private:
    float _invEnergyStep;
    float _interactionTableWidth;
    std::shared_ptr<BoundTexture3D<float, cudaArrayLayered>> _rayleighTables;
};
}
}
