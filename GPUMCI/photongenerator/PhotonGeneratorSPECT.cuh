#pragma once

#define BLOCKWIDTH 256

#include <odl_cpp_utils/cuda/disableThrustWarnings.h>
#include <thrust/device_vector.h>
#include <odl_cpp_utils/cuda/enableThrustWarnings.h>

#include <GPUMCI/physics/CudaSettings.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>

namespace gpumci {
namespace cuda {
namespace {
//  Randomizes the direction of the photon, form http://mathworld.wolfram.com/SpherePointPicking.html
template <typename Rng>
__device__ void
randomizeDirection(float3& direction, Rng& rng) {
    // Calculate the rotation angles
    float costheta = 2.0f * rng.rand() - 1.0f;
    float sintheta = sqrtf(1 - costheta * costheta);
    float cosphi, sinphi;
    sincospif(2.0f * rng.rand(), &sinphi, &cosphi);

    // Rotate the vector
    direction.x = costheta;
    direction.y = sintheta * sinphi;
    direction.z = sintheta * cosphi;
}
}

struct PhotonGeneratorSPECTSharedData {
};

struct PhotonGeneratorSPECTDevice {
    typedef PhotonGeneratorSPECTSharedData SharedData;
    typedef CudaMonteCarloScatterParticle Particle;

    __host__ PhotonGeneratorSPECTDevice(unsigned runs_per_photon)
        : _num_photons_left(runs_per_photon) {
    }

    __device__ void init(int idx, SharedData& sharedData, const CudaParameters& c_param) {
        int3 index{idx % c_param.volumeSize.x,
                   (idx / c_param.volumeSize.x) % c_param.volumeSize.y,
                   idx / (c_param.volumeSize.x * c_param.volumeSize.y)};

        _position = c_param.volumeMin + float3{index.x / c_param.inverseVoxelSize.x,
                                               index.y / c_param.inverseVoxelSize.y,
                                               index.z / c_param.inverseVoxelSize.z};
    }

    template <typename Rng>
    __device__ bool generatePhoton(Particle& photon, int idx, Rng& rng, SharedData& shared, const CudaParameters& c_param) {
        if (--_num_photons_left <= 0)
            return false;

        photon.position = _position + float3{rng.rand(), rng.rand(), rng.rand()} / c_param.inverseVoxelSize;
        randomizeDirection(photon.direction, rng);
        photon.weight = static_cast<float>(idx);
        photon.energy = 0.1f;
        return true;
    }

  private:
    float3 _position;
    unsigned _num_photons_left;
};

struct PhotonGeneratorSPECT {
    PhotonGeneratorSPECT(unsigned runs_per_photon)
        : _runs_per_photon{runs_per_photon} {
    }

    PhotonGeneratorSPECTDevice deviceSide() {
        return {_runs_per_photon};
    }

  private:
    const unsigned _runs_per_photon;
};
}
}
