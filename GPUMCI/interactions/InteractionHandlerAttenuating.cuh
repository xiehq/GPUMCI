/** @file
 *
 * Implementation of a monte carlo simulation for photons.
 *
 * Copyright Elekta Instrument AB, 2011
 */

#pragma once

#include <GPUMCI/physics/CudaSettings.h>
#include <GPUMCI/interactions/InteractionTypes.h>

#include <cuda_runtime.h>

namespace gpumci {
namespace cuda {

/**
	* TODO
	*/
struct InteractionHandlerAttenuating {
    __host__ InteractionHandlerAttenuating(const cudaTextureObject_t texMaterial)
        : _texMaterial(texMaterial) {}

    /**
	* Simulates an interaction
	*
	* Parameters:
	*		meanFreePathCM		The woodcock mean free path (in CM) at the energy of interaction
    *		stepCM              Length of the step taken
	*		myMedium			Index of the medium at the point of interaction
	*		myDensity			Density (g/cm^3) at the point of interaction
	*		ivoxel				Index of the interaction point
	*		primary				Boolean (which will be written to) indicating if the photon is a primary (non-scattered photon)
	*		rng					The random number generator to use.
	*/
    template <typename Rng, typename Particle>
    __device__ void simulateInteraction(const float meanFreePathCM,
                                        const float stepCM,
                                        const uint8_t medium,
                                        const float density,
                                        Particle& photon,
                                        Rng& rng,
                                        const CudaParameters& c_param) const {
        float att = getMassAttenuationCoefficient(medium,
                                                  photon.energy,
                                                  c_param);
        if (density == 0.0f)
            return;
        photon.weight *= expf(-density * att * stepCM);
        //        printf("ih %f %f %f %f %f %f %f %f %f %d\n",photon.energy,photon.weight,photon.position.x, photon.position.y, photon.position.z, photon.direction.x, photon.direction.y, photon.direction.z, myDensity,(int)myMedium);
    }

  private:
    //  For a given photon energy and material medium, reads the photon data
    //  textures and returns the mass attenuation coefficients
    //  for Compton scattering, photoelectric aborption, pair production, and
    //  Rayleigh scattering.
    __device__ float getMassAttenuationCoefficient(const uint8_t medium,
                                                   const float energy,
                                                   const CudaParameters& c_param) const {
        // Compute the array index. The '0.5f' in the energy offset causes the
        // code to round to the nearest energy, instead of
        // truncating and thus introducing a bias toward a lower energy.
        const float index = 0.5f + energy * c_param.invEnergyStep;
        const float mediumIndex = (medium) + 0.5f;

        float4 mu = tex2D<float4>(_texMaterial, index, mediumIndex);
        return mu.x + mu.y + mu.z;
    }

    // Todo, pack as float3
    const cudaTextureObject_t _texMaterial; // texture<float4,cudaTextureType2D,cudaReadModeElementType>
};
}
}
