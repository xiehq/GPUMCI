#pragma once

#include <GPUMCI/physics/CudaSettings.h>
#include <GPUMCI/utils/physical_constants.h>

namespace gpumci {
namespace cuda {
/**
	Method with computes a Compton Interaction using the method described in EVERETT-CASHWELL-1978.

	This method has no pre-requirements.
	*/
struct ComptonEverett {
    typedef ComptonEverett DeviceSideType;

    const DeviceSideType& deviceSide() const {
        return *this;
    }

    template <typename Rng, typename Particle>
    __device__ void operator()(const int myMedium, const Particle& myPhoton, float& costheta, float& energy, Rng& rng) const {
        // The following logic is taken verbatim from the flow chart on page 11 of reference EVERETT-CASHWELL-1978.

        // As explained in the introduction to the flow chart, alpha is the photon energy scaled by the electron rest mass.
        float alpha = myPhoton.energy * INVERSE_ELECTRON_REST_ENERGY_MEV;

        // Steps 1 to 3 of the flow chart.
        float eta = 1.0f + 2.0f * alpha;
        float xi = 1.0f / eta;
        float N = logf(eta);

        // As explained in the introduction to the flow chart, x is the fraction of the incident photon energy that remains
        // with the scattered photon.
        float x = 0.0f;

        // Steps 11 to 21 of the flow chart.
        float beta = 1.0f / alpha;
        float phi = 0.25f; //everettCashwellPhiFunction(alpha); Alpha is always "small"
        float x0 = xi + phi * (1.0f - xi);
        float M = logf(x0);
        float K1 = 1.0f - x0;
        float K2 = 1.0f / x0;
        float K3 = 1.0f - 2.0f * beta * (1.0f + beta);
        float F0 = K1 * (0.5f * (1.0f + x0) + beta * beta * (eta + K2)) - M * K3;
        float G = 2.0f * alpha * (1.0f + alpha) * xi * xi + 4.0f * beta + N * K3;
        float J0 = F0 / G;
        float r = rng.rand();

        if (r < J0) // Step 22 of the flow chart
        {
            // Step 23 of the flow chart.
            float R = r / J0;

            // Note: Steps 24 to 26 are absent from EVERETT-CASHWELL-1978. A note at the end
            // of the flow chart states that these steps were purposely omitted, while retaining
            // the numbering of the flow chart from a previous publication.

            // Steps 27 to 31 of the flow chart.
            float f0 = x0 + K2 + beta * beta * (1.0f - K2) * (eta - K2);
            float A0 = -0.5f * F0;
            float B0 = F0 + (F0 / f0) - 3.0f * K1;
            float C0 = A0 - (F0 / f0) + 2.0f * K1;

            x = 1.0f + R * (A0 + R * (B0 + R * C0));
        } else {
            // Steps 32 and 33 of the flow chart.
            float Lambda0 = (M + N) / (1.0f - J0);
            x = x0 * expf(-Lambda0 * (r - J0));
        }

        // As noted above, the scattered photon energy is x times the incident photon energy.
        energy = myPhoton.energy * x; // scattered photon energy

        // Now that the energy of the scattered photon is known, the photon scattering angle can be determined from kinematics.
        // The following expression can be derived from equation (2.8.8) of SLAC-265 or from equation (7.8) of ATTIX-1986. In a
        // slightly different form, it also appears on page 11 of EVERETT-CASHWELL-1978.
        costheta = 1.0f + ELECTRON_REST_ENERGY_MEV / myPhoton.energy - ELECTRON_REST_ENERGY_MEV / energy;
    }

  private:
    //  This is an auxiliary function called by function photonCompton. It implements the function phi(alpha) defined in Table I
    //  on page 8 of reference EVERETT-CASHWELL-1978.
    __device__ float everettCashwellPhiFunction(const float& alpha) const {
        // Per Table I, the value of phi is 0.25 for all alpha less than 0.962 and for all alpha greater than or equal to 10.0.
        float phi = 0.25f; // default value

        // Note: Technically, Table I does not assign any value to phi for alpha less than 0.002 or for alpha greater than or
        // equal to 202.0. The reasons are:
        //    a) the sampling method that uses function phi(alpha) is not intended to be used outside the energy range
        //        corresponding to alpha in the interval [0.002, 202.0);
        //    b) no error estimate |epsilon| is available for alpha outside the above interval.
        // This function would be an awkward place to enforce limits on alpha, however, and the energy range will be enforced
        // elswhere. Therefore, this function simply returns the default value of 0.25 for alpha below 0.002 or above 202.0.

        // The following lines of code implement rows 2 to 4 of Table I.
        if (alpha >= 0.962f && alpha < 1.642f) {
            phi = 0.20f;
        } else if (alpha >= 1.642f && alpha < 2.002f) {
            phi = 0.17f;
        } else if (alpha >= 2.002f && alpha < 10.0f) {
            phi = 0.15f;
        }

        return phi;
    }
};
}
}
