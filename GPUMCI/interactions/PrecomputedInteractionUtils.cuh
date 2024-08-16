#pragma once

#include <GPUMCI/interactions/PrecomputedInteractionUtils.h>

#include <odl_cpp_utils/cuda/disableThrustWarnings.h>
#include <odl_cpp_utils/cuda/enableThrustWarnings.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace gpumci {
namespace cuda {
namespace util {
inline thrust::device_vector<float> make_interaction_device(const InteractionTables& tables) {
    thrust::host_vector<float> host_data;
    unsigned n_data = tables.n_materials * tables.n_energies * tables.n_samples;
    host_data.reserve(n_data);

    for (unsigned mat = 0; mat < tables.n_materials; mat++) {
        for (unsigned i_energy = 0; i_energy < tables.n_energies; i_energy++) {
            for (unsigned i_sample = 0; i_sample < tables.n_samples; i_sample++) {
                double invcdf = tables.inverseCDF[mat](i_sample, i_energy);

                host_data.push_back(static_cast<float>(invcdf));
            }
        }
    }
    return {host_data}; // copies result to device
}
} // namespace util
} // namespace cuda
} // namespace gpumci
