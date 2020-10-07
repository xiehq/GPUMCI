#pragma once

#include <GPUMCI/physics/CudaSettings.h>

namespace gpumci {
namespace cuda {
/**
	Fill-in struct used to indicate that no interaction should happen.

	Optimizing compilers will fully remove the interaction.
	*/
struct NoInteraction {
    template <typename Rng, typename Particle>
    __device__ void operator()(const int myMedium, const Particle& myPhoton, float& costheta, float& energy, Rng& rng) const {
    }
};
}
}