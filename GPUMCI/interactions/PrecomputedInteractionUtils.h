#pragma once

#include <odl_cpp_utils/utils/cast.h>
#include <Eigen/Core>

namespace gpumci {

/*
 * InteractionTables
 * 
 * Contains a lookup-table of the inverse CDF for an interaction.
 * Parameters:
 *   energyStep     The distance (in MeV) between each entry of the tables.
 *   inverseCDF     Lookuptable. Each row should contain the values of the iCDF (as indexed from 0 to 1) and each column should contain a specific energy (starting at E=0). The elements of the vector each represent one material.
 */
struct InteractionTables {
    InteractionTables(double energyStep_,
                      const std::vector<Eigen::ArrayXXd>& inverseCDF_)
        : energyStep(energyStep_),
          n_energies(narrow_cast<unsigned>(inverseCDF_[0].cols())),
          n_samples(narrow_cast<unsigned>(inverseCDF_[0].rows())),
          n_materials(narrow_cast<unsigned>(inverseCDF_.size())),
          inverseCDF(inverseCDF_) {
        for (const auto& arr : inverseCDF) {
            if (arr.rows() != n_samples ||
                arr.cols() != n_energies) {
                throw std::invalid_argument("Sizes of inverse CDF's have to match");
            }
        }
    }

    const double energyStep;
    const unsigned n_energies;
    const unsigned n_samples;
    const unsigned n_materials;
    const std::vector<Eigen::ArrayXXd> inverseCDF;
};
}
