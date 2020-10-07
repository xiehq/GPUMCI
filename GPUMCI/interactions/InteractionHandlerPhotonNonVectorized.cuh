#pragma once

#include <GPUMCI/physics/CudaSettings.h>
#include <GPUMCI/interactions/InteractionTypes.h>
#include <LCRUtils/cuda/cutil_math.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace gpumci {
namespace cuda {
namespace {

//  Samples randomly from a discrete probability distribution to decide what
//  type of interaction a photon will undergo.
template <typename Rng>
__device__ InteractionType
whichPhotonInteraction(const float meanFreePath, const float density,
                       const float muCompton, const float muPhoto,
                       const float muRayleigh, Rng& rng) {
    // The mean free path of the particle is 1/muTotal, where muTotal =
    // muCompton + muPhoto + muPair + muRayleigh + muFictitious.
    // Each mu here is a linear attenuation coefficient (not a mass attenuation
    // coefficient). The probability of any given
    // interaction is equal to muInteraction / muTotal = muInteraction *
    // meanFreePath.

    // We now sample from the discrete distribution defined by the probabilities
    // computed above. Note that we gain efficiency
    // by putting the Compton interaction first, inasmuch as this interaction
    // has the highest mass attenuation coefficient for
    // water and similar materials at radiotherapy energies. Putting the most
    // probable interaction first allows fewer additions
    // to be done and fewer conditions to be checked.
    float scale = meanFreePath * density;
    if (scale <= 0.0f) return InteractionType::INTERACTION_FICTIOUS;

    float rn = rng.rand() / scale;

    float probabilitySum = muPhoto;
    if (rn <= probabilitySum) {
        return InteractionType::INTERACTION_PHOTO;
    }

    probabilitySum += muCompton;
    if (rn <= probabilitySum) {
        return InteractionType::INTERACTION_COMPTON;
    }

    probabilitySum += muRayleigh;
    if (rn <= probabilitySum) {
        return InteractionType::INTERACTION_RAYLEIGH;
    }

    return InteractionType::INTERACTION_FICTIOUS;
}

//  direction vector gets rotated (in place) by theta and U(0,2pi) phi
template <typename Rng>
__device__ void
rotateDirectionVector(float3& direction, float costheta, Rng& rng) {
    // Calculate the rotation angles
    costheta = clamp(costheta, -1.0f, 1.0f); // To avoid numerical issues in sqrt
    float sintheta = sqrtf(1 - costheta * costheta);
    float cosphi, sinphi;
    sincospif(2.0f * rng.rand(), &sinphi, &cosphi);

    // Rotate the vector
    float sqrtz_inv = rsqrtf(1.0f - direction.z * direction.z);
    float oldx = direction.x, oldy = direction.y, oldz = direction.z;

    direction.x = oldx * costheta +
                  (oldx * oldz * cosphi - oldy * sinphi) * sintheta * sqrtz_inv;
    direction.y = oldy * costheta +
                  (oldy * oldz * cosphi + oldx * sinphi) * sintheta * sqrtz_inv;
    direction.z = oldz * costheta - sintheta * cosphi / sqrtz_inv;

    direction = normalize(direction);
}
}

/**
 * Struct which handles interactions according to pre-selected
 * interaction handlers `Compton`, `Rayleigh` and `Photo`.
 */
template <typename Compton, typename Rayleigh, typename Photo>
struct InteractionHandlerPhotonNonVectorized {
    __host__ InteractionHandlerPhotonNonVectorized(const Compton &compton,
                                                   const Rayleigh &rayleigh,
                                                   const Photo &photo,
                                                   const cudaTextureObject_t &texCompton,
                                                   const cudaTextureObject_t &texRayleigh,
                                                   const cudaTextureObject_t &texPhoto) // texture<float4,cudaTextureType2D,cudaReadModeElementType>
        : _compton(compton),
          _rayleigh(rayleigh),
          _photo(photo),
          _texCompton(texCompton),
          _texRayleigh(texRayleigh),
          _texPhoto(texPhoto) {}

    /**
     * Simulates an interaction
     *
     * Parameters:
     *	meanFreePathCM		The woodcock mean free path (in CM) at the energy of interaction
     *	medium			    Index of the medium at the point of interaction
     *	density			    Density (g/cm^3) at the point of interaction
     *	primary			    Boolean (which will be written to) indicating if the photon is a primary (non-scattered photon)
     *	rng                 The random number generator to use.
     */
    template <typename Rng>
    __device__ void simulateInteraction(const float meanFreePathCM,
                                        const uint8_t medium,
                                        const float density, bool& primary,
                                        CudaMonteCarloParticle& photon,
                                        Rng& rng,
                                        const CudaParameters& c_param) const {
        float muCompton, muPhoto, muRayleigh; // Return parameters
        getMassAttenuationCoefficientsPhoton(medium, photon.energy, muCompton, muPhoto, muRayleigh, c_param);

        // Sample the interaction type.
        InteractionType interactionType = whichPhotonInteraction(meanFreePathCM, density, muCompton, muPhoto, muRayleigh, rng);

        // Energy, polar angle, and azimuthal angle for the scattered photon.
        float energy = photon.energy;
        float costheta = 1.0f;

        switch (interactionType) {
        case InteractionType::INTERACTION_COMPTON:
            primary = false;
            _compton(medium, photon, costheta, energy, rng);
            break;
        case InteractionType::INTERACTION_PHOTO:
            primary = false;
            _photo(medium, photon, costheta, energy, rng);
            break;
        case InteractionType::INTERACTION_RAYLEIGH:
            primary = false;
            _rayleigh(medium, photon, costheta, energy, rng);
            break;
        case InteractionType::INTERACTION_FICTIOUS:
            break;
        default:
            assert(false); // This should not happen
            break;
        } // end switch(interactionType)

        // Set the photon to its post-interaction energy and direction.
        if (energy >= c_param.photon_energy_cutoff) {
            photon.energy = energy;
            rotateDirectionVector(photon.direction, costheta, rng);
        } else {
            photon.energy = 0.0f;
        }
    }

  private:
    //  For a given photon energy and material medium, reads the photon data
    //  textures and returns the mass attenuation coefficients
    //  for Compton scattering, photoelectric aborption, pair production, and
    //  Rayleigh scattering.
    __device__ void getMassAttenuationCoefficientsPhoton(
        const uint8_t medium, const float energy, float& muCompton,
        float& muPhoto, float& muRayleigh,
        const CudaParameters& c_param) const {
        // Compute the array index. The '0.5f' in the energy offset causes the
        // code to round to the nearest energy, instead of
        // truncating and thus introducing a bias toward a lower energy.
        const float index = 0.5f + energy * c_param.invEnergyStep;
        const float mediumIndex = (medium) + 0.5f;

        muCompton = tex2D<float>(_texCompton, index, mediumIndex);
        muRayleigh = tex2D<float>(_texRayleigh, index, mediumIndex);
        muPhoto = tex2D<float>(_texPhoto, index, mediumIndex);
    }

    const Compton _compton;
    const Rayleigh _rayleigh;
    const Photo _photo;

    const cudaTextureObject_t _texCompton;  // texture<float,cudaTextureType2D,cudaReadModeElementType>
    const cudaTextureObject_t _texRayleigh; // texture<float,cudaTextureType2D,cudaReadModeElementType>
    const cudaTextureObject_t _texPhoto;    // texture<float,cudaTextureType2D,cudaReadModeElementType>
};

/**
 * Creates an PhotonInteractionHandler with given parameters.
 */
template <typename Compton, typename Rayleigh, typename Photo>
InteractionHandlerPhotonNonVectorized<Compton, Rayleigh, Photo>
makePhotonInteractionHandlerNonVectorized(const Compton& compton,
                                          const Rayleigh& rayleigh,
                                          const Photo& photo,
                                          const cudaTextureObject_t& texCompton,
                                          const cudaTextureObject_t& texRayleigh,
                                          const cudaTextureObject_t& texPhoto) // texture<float4,cudaTextureType2D,cudaReadModeElementType>
{
    return {compton, rayleigh, photo, texCompton, texRayleigh, texPhoto};
}
}
}
