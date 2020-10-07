#include <GPUMCI/implementations/MaterialUtils.cuh>
#include <odl_cpp_utils/cuda/cutil_math.h>

#include <stdexcept>

namespace gpumci {
namespace cuda {
namespace util {
//Creates a device_vector with the data in materials
thrust::device_vector<float4> make_material_device(const MaterialData& materials) {
    thrust::host_vector<float4> host_data;

    for (const auto& table : {materials.compton, materials.rayleigh, materials.photo}) {
        if (table.rows() != materials.n_materials ||
            table.cols() != materials.n_energies)
            throw std::invalid_argument("data sizes of the material tables has to match");
    }

    for (unsigned mat = 0; mat < materials.n_materials; mat++) {
        for (unsigned i_energy = 0; i_energy < materials.n_energies; i_energy++) {
            host_data.push_back(make_float4(static_cast<float>(materials.compton(mat, i_energy)),
                                            static_cast<float>(materials.rayleigh(mat, i_energy)),
                                            static_cast<float>(materials.photo(mat, i_energy)),
                                            0.0f));
        }
    }
    return {host_data.begin(), host_data.end()};
}
}
}
}