#pragma once

#include <odl_cpp_utils/cuda/disableThrustWarnings.h>
#include <odl_cpp_utils/cuda/enableThrustWarnings.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <GPUMCI/physics/MaterialEntry.h>

namespace gpumci {
namespace cuda {
namespace util {
thrust::device_vector<float> make_woodcock_tables(const float* densityDevice,
                                                  const uint8_t* materialTypeDevice,
                                                  const int3 volumeSize,
                                                  const MaterialData& attenuationData);
}
} // namespace cuda
} // namespace gpumci
