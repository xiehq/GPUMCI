#pragma once

#include "cuda_runtime.h"
#include <odl_cpp_utils/cuda/CuVector.h>

namespace gpumci {
namespace cuda {

struct CudaMonteCarloParticle {
    CudaMonteCarloParticle() = default;

    CudaMonteCarloParticle(const Eigen::Vector3d& position_,
                           const Eigen::Vector3d& direction_,
                           double energy,
                           double weight)
        : position(make_float3(position_)),
          direction(make_float3(direction_)),
          energy(static_cast<float>(energy)),
          weight(static_cast<float>(weight)) {
        assert(energy >= 0.0f);
        assert(norm(direction) > 0.999f);
        assert(norm(direction) < 1.001f);
    }

    CudaMonteCarloParticle(const Eigen::Vector3d& position_,
                           const Eigen::Vector3d& direction_,
                           double energy,
                           double weight,
                           int    data
                          )
    : position(make_float3(position_)),
    direction(make_float3(direction_)),
    energy(static_cast<float>(energy)),
    weight(static_cast<float>(weight)),
    data(data){
        assert(energy >= 0.0f);
        assert(norm(direction) > 0.999f);
        assert(norm(direction) < 1.001f);
    }
    

    void setPosition(const Eigen::Vector3f& position_)
    {
      position = make_float3(position_);
    }

    void
    setPosition(const Eigen::Vector3f& position_) {
        position = make_float3(position_);
    }

    void setDirection(const Eigen::Vector3f& direction_) {
        direction = make_float3(direction_);
    }

    float3 position;
    float3 direction;
    float energy;
    float weight;
    int   data;
};

struct CudaMonteCarloScatterParticle : CudaMonteCarloParticle {
    CudaMonteCarloScatterParticle() = default;

    CudaMonteCarloScatterParticle(const Eigen::Vector3d& position_,
                                  const Eigen::Vector3d& direction_,
                                  double energy,
                                  double weight,
                                  bool primary = true)
        : CudaMonteCarloParticle(position_,
                                 direction_,
                                 energy,
                                 weight),
          primary(primary) {}
        

    __device__ CudaMonteCarloScatterParticle& operator=(const CudaMonteCarloParticle& other) {
        CudaMonteCarloParticle::operator=(other);
        primary = true;
        return *this;
    }

    __device__ CudaMonteCarloScatterParticle& operator=(const CudaMonteCarloScatterParticle& other) {
        CudaMonteCarloParticle::operator=(other);
        primary = other.primary;
        return *this;
    }

    int  data;
    bool primary;
};

struct CudaParameters {
    CudaParameters(const Eigen::Vector3i& volumeSize_,
                   const Eigen::Vector3d& volumeOrigin,
                   const Eigen::Vector3d& voxelSize,
                   const double energyStep,
                   const double photon_energy_cutoff_ = 0.01)
        : volumeSize(make_int3(volumeSize_)),
          volumeMin(make_float3(volumeOrigin)),
          volumeMax(make_float3(static_cast<float>(volumeOrigin[0] + volumeSize_[0] * voxelSize[0]),
                                static_cast<float>(volumeOrigin[1] + volumeSize_[1] * voxelSize[1]),
                                static_cast<float>(volumeOrigin[2] + volumeSize_[2] * voxelSize[2]))),
          inverseVoxelSize(make_float3(1.0f / (float)voxelSize[0],
                                       1.0f / (float)voxelSize[1],
                                       1.0f / (float)voxelSize[2])),
          invEnergyStep{static_cast<float>(1.0 / energyStep)},
          photon_energy_cutoff(static_cast<float>(photon_energy_cutoff_)) {}

    // Volume
    const int3 volumeSize;
    const float3 volumeMin;
    const float3 volumeMax;
    const float3 inverseVoxelSize;

    // Misc parameters
    const float invEnergyStep;
    const float photon_energy_cutoff;
};
} // namespace cuda
} // namespace gpumci
