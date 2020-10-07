#include <GPUMCI/implementations/WoodcockUtils.cuh>

#include <odl_cpp_utils/cuda/disableThrustWarnings.h>
#include <thrust/functional.h>
#include <odl_cpp_utils/cuda/enableThrustWarnings.h>

#include <odl_cpp_utils/utils/cast.h>
#include <odl_cpp_utils/cuda/cutil_math.h>

namespace gpumci {
namespace cuda {
namespace {
//Calculates the woodcock step length with a set of materials and densities
thrust::device_vector<float> make_woodcock_device(const MaterialData& attenuationData,
                                                  const std::vector<float>& maxDensities) {
    assert(maxDensities.size() <= attenuationData.n_materials);
    thrust::host_vector<float> host_data;

    for (unsigned i_energy = 0; i_energy < attenuationData.n_energies; i_energy++) {
        double minMfp = 1000.0;
        for (unsigned mat = 0; mat < maxDensities.size(); mat++) {
            double att = maxDensities[mat] * (attenuationData.compton(mat, i_energy) +
                                              attenuationData.rayleigh(mat, i_energy) +
                                              attenuationData.photo(mat, i_energy));
            if (att > 0.01 && att < 100.0) //We ignore extreme values (caused by e.g. 0 densities)
                minMfp = std::min(minMfp, 1.0f / att);
        }
        host_data.push_back(static_cast<float>(minMfp));
    }
    return {host_data}; //copy result to device
}

template <typename T>
__host__ __device__ T cumax(T a, T b) {
    return a > b ? a : b;
}

//Helper used to determine the maximum element with a given key
template <typename Key, typename Value>
struct maxByKey {
    const Key _key;
    __host__ maxByKey(Key key) : _key(key) {}

    __host__ __device__
        thrust::tuple<Key, Value>
        operator()(const thrust::tuple<Key, Value>& lhs,
                   const thrust::tuple<Key, Value>& rhs) const {
        if (thrust::get<0>(lhs) == _key && thrust::get<0>(rhs) == _key)
            return {_key, cumax(thrust::get<1>(lhs), thrust::get<1>(rhs))};
        else if (thrust::get<0>(lhs) == _key)
            return lhs;
        else
            return rhs;
    }
};
}
namespace util {
thrust::device_vector<float> make_woodcock_tables(const float* densityDevice,
                                                  const uint8_t* materialTypeDevice,
                                                  const int3 volumeSize,
                                                  const MaterialData& attenuationData) {
    //Find max density per material
    auto nel = volumeSize.x * volumeSize.y * volumeSize.z;

    //Create some iterators
    auto mat_first = thrust::device_pointer_cast(materialTypeDevice);
    auto mat_last = mat_first + nel;
    auto dens_first = thrust::device_pointer_cast(densityDevice);
    auto zip_first = thrust::make_zip_iterator(thrust::make_tuple(mat_first, dens_first));
    auto zip_last = zip_first + nel;

    //Find number of elements
    uint8_t max_element = thrust::reduce(mat_first, mat_last, 0, thrust::maximum<uint8_t>{});
    std::vector<float> max_densities(attenuationData.n_materials);

    if (max_element >= attenuationData.n_materials) {
        throw std::invalid_argument("Max material index larger than n_materials in database");
    }

    //Find max density per element
    for (uint8_t el = 0; el <= max_element; el++) {
        max_densities[el] = thrust::get<1>(thrust::reduce(zip_first,
                                                          zip_last,
                                                          thrust::make_tuple(el, 0.0f),
                                                          maxByKey<uint8_t, float>{el}));
    }

    //Create woodcock table
    int n_energies = narrow_cast<int>(attenuationData.n_energies);
    return make_woodcock_device(attenuationData, max_densities);
}
}
}
}
