#pragma once

#include <odl_cpp_utils/cuda/disableThrustWarnings.h>
#include <thrust/device_vector.h>
#include <odl_cpp_utils/cuda/enableThrustWarnings.h>

#include <GPUMCI/physics/MaterialEntry.h>

namespace gpumci {
namespace cuda {
namespace util {
//Creates a device_vector with the data in materials
thrust::device_vector<float4> make_material_device(const MaterialData& materials);
}
}
}