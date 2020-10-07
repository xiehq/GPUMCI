#pragma once

#include <GPUMCI/physics/CudaSettings.h>

namespace gpumci {
namespace cuda {

//  This function simulates photoelectric absorption. The photoelectron receives all the energy of the photon,
//  and the photon is completely absorbed. The photoelectron is also assumed to be absorbed instantly.
struct PhotonPhoto {
    template <typename Rng, typename Particle>
    __device__ void operator()(const int myMedium, const Particle& myPhoton, float& costheta, float& energy, Rng& rng) const {
        energy = 0;
    }
};

}
}