/** @file
 *
 * Implementation of a monte carlo simulation for photons.
 *
 * Copyright Elekta Instrument AB, 2011
 */

#pragma once

#include <GPUMCI/interactions/InteractionTypes.h>
#include <GPUMCI/physics/CudaSettings.h>

#include <cuda_runtime.h>

namespace gpumci {
namespace cuda {

/**
 * TODO
 */
struct InteractionHandlerProjector {
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
        photon.weight *= expf(-stepCM / meanFreePathCM);

        //        printf("ih %f %f %f %f %f %f %f %f %f %d\n",photon.energy,photon.weight,photon.position.x, photon.position.y, photon.position.z, photon.direction.x, photon.direction.y, photon.direction.z, myDensity,(int)myMedium);
    }
};
} // namespace cuda
} // namespace gpumci
